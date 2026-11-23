"""Training-loss curves of the released open-sci-ref-0.01 runs from their Megatron slurm logs.

    python3 -I scripts/opensciref/parse_logs.py LOGDIR CFGDIR
      LOGDIR  extracted open-sci/open-sci-ref-0.01-logs/logs.tar.gz
      CFGDIR  <size>-<corpus>/config.json of each released repo (its _name_or_path names the source run and job)
    -> results/curve_forecast/opensciref/train_logs.json

Per run: the 'lm loss' (nats per token, averaged by Megatron over each 50-iteration log interval) at every logged
iteration, the final validation/test losses, and the launch arguments that matter (batch, iterations, warmup, WSD
decay, data path, split). Restarts inside one log are stitched by keeping the last value logged for an iteration.
"""
import glob
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results/curve_forecast/opensciref/train_logs.json")
IT = re.compile(r"iteration\s+(\d+)/\s*(\d+) \| consumed samples:\s+(\d+) .*?learning rate: ([0-9.E+-]+) \| global batch size:\s+(\d+) \| lm loss: ([0-9.E+-]+)")
VAL = re.compile(r"validation loss at iteration (\d+)(?: on (validation|test) set)? \| lm loss value: ([0-9.E+-]+)")
ARGS = {k: re.compile(r"--%s (\S+)" % k) for k in ("global-batch-size", "seq-length", "train-iters", "lr-warmup-iters",
                                                    "lr-wsd-decay-iters", "lr", "data-path", "split", "tokenizer-model",
                                                    "num-layers", "hidden-size", "eval-iters", "eval-interval")}


def parse(path):
    it, val, args = {}, {}, {}
    with open(path, errors="replace") as f:
        for line in f:
            m = IT.search(line)
            if m:
                i = int(m.group(1))
                it[i] = dict(lm_loss=float(m.group(6)), lr=float(m.group(4)), consumed_samples=int(m.group(3)),
                             global_batch=int(m.group(5)), train_iters=int(m.group(2)))
                continue
            m = VAL.search(line)
            if m:
                val[f"{m.group(2) or 'default'}@{m.group(1)}"] = float(m.group(3))
                continue
            if "pretrain_gpt.py" in line and not args:
                for k, r in ARGS.items():
                    mm = r.search(line)
                    if mm:
                        args[k] = mm.group(1)
    return it, val, args


def main():
    logdir, cfgdir = sys.argv[1], sys.argv[2]
    files = {os.path.basename(p): p for p in glob.glob(os.path.join(logdir, "**", "*.out"), recursive=True)}
    out = {}
    for cfg in sorted(glob.glob(os.path.join(cfgdir, "*", "config.json"))):
        key = os.path.basename(os.path.dirname(cfg))
        run = json.load(open(cfg))["_name_or_path"].split("/")[-3]
        final_iter = json.load(open(cfg))["_name_or_path"].split("/")[-1]
        hits = [p for n, p in files.items() if n.startswith(run)]
        assert len(hits) == 1, (key, hits)
        it, val, args = parse(hits[0])
        its = sorted(it)
        out[key] = dict(run=run, log=os.path.basename(hits[0]), main_branch_iter=final_iter, args=args,
                        first_iter=its[0], last_iter=its[-1], n_logged=len(its), validation=val,
                        iters=its, lm_loss=[it[i]["lm_loss"] for i in its], lr=[it[i]["lr"] for i in its],
                        consumed_samples=[it[i]["consumed_samples"] for i in its])
        print(key, its[0], its[-1], len(its), args.get("global-batch-size"), args.get("train-iters"),
              args.get("lr-warmup-iters"), args.get("lr-wsd-decay-iters"), args.get("split"), val, flush=True)
    json.dump(out, open(OUT, "w"))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
