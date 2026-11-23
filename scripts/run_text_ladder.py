"""Run train_text.py over tokenizers x byte budgets x seeds, skipping existing outputs."""
import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ap = argparse.ArgumentParser()
ap.add_argument("--toks", default="bytes,bpe512,bpe2048,bpe8192")
ap.add_argument("--log2_bytes", default="20,22,24,26")
ap.add_argument("--seeds", default="0")
ap.add_argument("--tag", default="v1")
ap.add_argument("--extra", default="", help='passed through, e.g. "--device cuda"')
args = ap.parse_args()

out = HERE.parent / "results" / "text_runs" / args.tag


def fingerprint(cmd):
    """Provenance fingerprint of the run this command would make (tok_dir runs only)."""
    r = subprocess.run(cmd + ["--fingerprint_only"], check=True, capture_output=True, text=True)
    return r.stdout.strip().splitlines()[-1]


for k in map(int, args.log2_bytes.split(",")):
    for seed in map(int, args.seeds.split(",")):
        for tok in args.toks.split(","):
            cmd = [sys.executable, str(HERE / "train_text.py"), "--tok", tok, "--train_bytes", f"2^{k}",
                   "--seed", str(seed), "--tag", args.tag] + shlex.split(args.extra)
            f = out / f"{tok}_b{k}_s{seed}.json"
            if f.exists():
                if "--tok_dir" not in cmd:  # TinyStories runs: unchanged behaviour, an existing output is final
                    continue
                if json.loads(f.read_text()).get("provenance", {}).get("fingerprint") == fingerprint(cmd):
                    continue  # exact match: trainer, arguments, tokenizer and data checksums
                stale = f.with_suffix(f".stale{len(list(out.glob(f.stem + '.stale*')))}")
                f.rename(stale)
                print(f"existing {f.name} has a different provenance; moved to {stale.name} and rerun", flush=True)
            print(" ".join(cmd), flush=True)
            subprocess.run(cmd, check=True)
