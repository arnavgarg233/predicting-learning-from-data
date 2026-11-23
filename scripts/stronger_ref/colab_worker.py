"""Colab worker: Qwen2.5-1.5B reference curves on a seeded 1,000-target subset.

Run from the bundle root (scripts/, text/refcurve/, jobs.json):  python -u colab_worker.py  (resumes; appends out/results.jsonl)

Per job: prep_corpus.py --no_tok with the stored arguments (one job prepared ahead in a thread), the reference partition
checked against the stored sha256 pair, target selection reproduced with refcurve_fineweb.run_model's code (rng seed 0,
nt = min(20000, eligible positions)), the stored subset checksum verified, then the model's NLL at the 18 stored contexts on
the subset, summarised with refcurve_fineweb.summarize. On the first job also: Qwen2.5-0.5B fp16 on the subset (to compare
with its stored fp32 losses) and Qwen2.5-1.5B fp32 on 200 subset targets (dtype check).
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import unicodedata  # noqa: F401  (used by refcurve_fineweb.load_docs)

import numpy as np
import torch

sys.path.insert(0, "text/refcurve")
import refcurve_fineweb as R  # noqa: E402

J = json.load(open("jobs.json"))
SUB_SEED, N_SUB = J["subset_seed"], J["n_sub"]
NS = R.NS
OUT = os.environ.get("OUT", "out")
ORDER = os.environ.get("ORDER", "fwd")   # "rev": test80 jobs in reverse order only (a second Colab tab)
AHEAD = int(os.environ.get("AHEAD", "2"))
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B")   # model switch: e.g. MODEL=Qwen/Qwen2.5-7B TAG=7b (RunPod 2026-10-26)
TAG = os.environ.get("TAG", "1p5b")                    # record key rows_<TAG>_fp16, npz <name>_<TAG>.npz
FP32_OFFLOAD = os.environ.get("FP32_OFFLOAD") == "1"   # fp32 dtype check with CPU offload (7B fp32 does not fit 24 GB)
FP32_GPU_MEM = os.environ.get("FP32_GPU_MEM", "16GiB")   # GPU share of the offloaded fp32 check (14B on 48 GB: larger)
DCHECK_N = int(os.environ.get("DCHECK_N", "200"))      # targets in the fp32 dtype check (14B run: 100)
SUB_N = int(os.environ.get("SUB_N", "0"))   # >0: only a seeded random SUB_N of the 1,000 subset targets (14B fallback: 500)
os.makedirs(OUT, exist_ok=True)
LOG = open(f"{OUT}/log.txt", "a")
DEV = torch.device("cuda")
CHUNK = int(os.environ.get("CHUNK", "100"))   # 7B fp16 on 24 GB: CHUNK=50 (batch grouping only; same targets and contexts)


def log(*a):
    s = time.strftime("%H:%M:%S ") + " ".join(map(str, a))
    print(s, flush=True)
    LOG.write(s + "\n")
    LOG.flush()


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def prep(j):
    d = f"data/{j['name']}"
    if os.path.exists(f"{d}/OK"):
        return True, "cached"
    t0 = time.time()
    cmd = f"{sys.executable} -u {os.environ.get('PREP_SCRIPT', 'scripts/prep_corpus.py')} --corpus {j['prep_corpus']} --out {d} {j['prep_extra']} --no_tok"
    with open(f"{OUT}/prep_{j['name']}.log", "w") as f:
        rc = subprocess.run(["bash", "-c", cmd], stdout=f, stderr=subprocess.STDOUT, timeout=5400).returncode
    if rc != 0:
        return False, f"prep rc {rc}"
    got = [sha(f"{d}/ref_stats_bytes.npy"), sha(f"{d}/ref_stats_docstarts.npy")]
    for fn in os.listdir(d):
        if not fn.startswith("ref_stats_") or fn.endswith("_gpt2.npy"):
            p = os.path.join(d, fn)
            shutil.rmtree(p) if os.path.isdir(p) else os.remove(p)
    if got != j["corpus_sha"]:
        return False, f"sha mismatch {got[0][:12]} vs {j['corpus_sha'][0][:12]}"
    open(f"{d}/OK", "w").write(f"{time.time() - t0:.0f}")
    return True, f"{time.time() - t0:.0f}s"


MODELS = {}


def model(name, dt, offload=False):
    k = (name, dt)
    if k not in MODELS:
        for kk in list(MODELS):
            del MODELS[kk]
        torch.cuda.empty_cache()
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if offload:
            m = AutoModelForCausalLM.from_pretrained(name, dtype=dt, device_map="auto", max_memory={0: FP32_GPU_MEM, "cpu": "200GiB"}).eval()
        else:
            try:
                m = AutoModelForCausalLM.from_pretrained(name, dtype=dt)
            except TypeError:
                m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dt)
            m = m.to(DEV).eval()
        MODELS[k] = (AutoTokenizer.from_pretrained(name), m)
    return MODELS[k]


def targets(j, tok, docs):
    """Exactly refcurve_fineweb.run_model's target selection, then the seeded subset."""
    u2b = {v: k for k, v in R.bytes_to_unicode().items()}
    V = max(len(tok), 152064)
    tb = np.zeros(V, np.int32)
    for t, i in tok.get_vocab().items():
        tb[i] = len(t) if all(c in u2b for c in t) else len(t.encode("utf-8"))
    ids = [tok(d, add_special_tokens=False)["input_ids"] for d in docs]
    for d, x in zip(docs, ids):
        assert int(tb[x].sum()) == len(d.encode("utf-8")), "token byte lengths do not add up"
    nmax = max(NS)
    ok_pos = np.array([max(len(x) - nmax, 0) for x in ids])
    rng = np.random.default_rng(0)
    nt = min(20000, int(ok_pos.sum()))
    flat = np.sort(rng.choice(int(ok_pos.sum()), nt, replace=False))
    cs = np.cumsum(ok_pos)
    doc = np.searchsorted(cs, flat, side="right")
    pos = flat - (cs[doc] - ok_pos[doc]) + nmax
    tgt = np.array([ids[d][p] for d, p in zip(doc, pos)])
    assert nt == j["nt"], f"nt {nt} vs stored {j['nt']}"
    idx = np.sort(np.random.default_rng(SUB_SEED).choice(nt, min(N_SUB, nt), replace=False))
    s = hashlib.sha256(tgt[idx].astype(np.int64).tobytes()).hexdigest()[:16]
    assert s == j["tgt_sub_sha"], f"subset target checksum {s} vs stored {j['tgt_sub_sha']}"
    doc, pos, tgt = doc[idx], pos[idx], tgt[idx]
    ctxb = np.array([[tb[ids[d][p - n:p]].sum() for n in NS] for d, p in zip(doc, pos)])
    return ids, doc, pos, tgt, tb[tgt].astype(float), ctxb


def nll(m, ids, doc, pos, tgt, chunk=CHUNK):
    loss = np.zeros((len(doc), len(NS)), np.float32)
    for c in range(0, len(doc), chunk):
        sl = slice(c, min(c + chunk, len(doc)))
        for k, n in enumerate(NS):
            x = torch.tensor([ids[d][p - n:p] for d, p in zip(doc[sl], pos[sl])], device=DEV)
            with torch.no_grad():
                h = m.base_model(x).last_hidden_state[:, -1]
                lp = torch.log_softmax(m.get_output_embeddings()(h).float(), -1)
            loss[sl, k] = -lp[torch.arange(len(x), device=DEV), torch.as_tensor(tgt[sl], device=DEV)].cpu().numpy()
    assert np.isfinite(loss).all(), "non-finite loss"
    return loss


def rows(loss, doc, tbt, ctxb, n_docs, label):
    r = R.summarize(label, label, loss / np.log(2), doc, tbt, ctxb, NS, n_docs)
    return [[x["n_tokens"], round(x["bits_per_byte"], 5), round(x["bpb_se"], 5), round(x["context_bytes_exact_mean"], 3)] for x in r["rows"]]


def run(j, first):
    d = f"data/{j['name']}"
    docs = R.load_docs(d, 4 * max(NS))
    tok, m = model(MODEL, torch.float16)
    ids, doc, pos, tgt, tbt, ctxb = targets(j, tok, docs)
    k = np.arange(len(doc))
    if SUB_N:   # positions within the sorted 1,000-target subset; same seed for every corpus
        k = np.sort(np.random.default_rng(20261027).choice(len(doc), min(SUB_N, len(doc)), replace=False))
        doc, pos, tgt, tbt, ctxb = doc[k], pos[k], tgt[k], tbt[k], ctxb[k]
    t0 = time.time()
    L = nll(m, ids, doc, pos, tgt)
    rec = dict(name=j["name"], group=j["group"], n_sub=len(doc), n_docs_total=len(docs), seconds=round(time.time() - t0),
               **{f"rows_{TAG}_fp16": rows(L, doc, tbt, ctxb, len(docs), MODEL.split("/")[-1])})
    np.savez_compressed(f"{OUT}/{j['name']}_{TAG}.npz", loss=L, doc=doc, pos=pos, tgt=tgt, k=k)
    if first:
      try:
        tok5, m5 = model("Qwen/Qwen2.5-0.5B", torch.float16)
        L5 = nll(m5, ids, doc, pos, tgt)
        rec["rows_0p5b_fp16"] = rows(L5, doc, tbt, ctxb, len(docs), "Qwen2.5-0.5B")
        np.savez_compressed(f"{OUT}/{j['name']}_0p5b_fp16.npz", loss=L5)
        del m, m5
        tok, m32 = model(MODEL, torch.float32, offload=FP32_OFFLOAD)
        s = slice(0, DCHECK_N)
        L32 = nll(m32, ids, doc[s], pos[s], tgt[s], chunk=20)
        np.savez_compressed(f"{OUT}/{j['name']}_{TAG}_fp32_{DCHECK_N}.npz", loss=L32)
        rec[f"dtype_check_{DCHECK_N}"] = dict(fp32=rows(L32, doc[s], tbt[s], ctxb[s], len(docs), "fp32"),
                                      fp16=rows(L[s], doc[s], tbt[s], ctxb[s], len(docs), "fp16"),
                                      mean_abs_nat_diff=float(np.abs(L32 - L[s]).mean()))
      except Exception as e:  # noqa: BLE001
        rec["check_error"] = f"{e!r}"[:200]
      for k in list(MODELS):
        del MODELS[k]
      torch.cuda.empty_cache()
    return rec


def main():
    rp = f"{OUT}/results.jsonl"
    done = {r["name"] for r in map(json.loads, open(rp)) if "error" not in r} if os.path.exists(rp) else set()
    for extra in os.environ.get("SKIP_FILES", "").split(","):
        if extra and os.path.exists(extra):
            done |= {r["name"] for r in map(json.loads, open(extra)) if "error" not in r}
    jobs = J["jobs"] if ORDER == "fwd" else [j for j in J["jobs"] if j["group"] == "test80"][::-1]
    todo = [j for j in jobs if j["name"] not in done]
    only = [x for x in os.environ.get("ONLY", "").split(",") if x]   # RunPod smoke test: run just these names
    if only:
        todo = [j for j in J["jobs"] if j["name"] in only]
    last = os.environ.get("LAST", "").split(",")        # slow preparations (wiki_ceb) go to the end
    todo = [j for j in todo if j["name"] not in last] + [j for j in todo if j["name"] in last]
    log("start", len(todo), "jobs to do;", len(done), "done;", torch.cuda.get_device_name(0))
    res = {}

    def prep_thread():
        for j in todo:
            while sum(1 for k in res if res[k] is not None and k not in finished) >= AHEAD:
                time.sleep(5)
            try:
                res[j["name"]] = prep(j)
            except Exception as e:  # noqa: BLE001
                res[j["name"]] = (False, f"prep exception {e!r}"[:300])
            log("prep", j["name"], res[j["name"]])

    finished = set()
    threading.Thread(target=prep_thread, daemon=True).start()
    state = dict(first=(not done and ORDER == "fwd") or os.environ.get("FORCE_FIRST") == "1")   # FORCE_FIRST: hardware checks
    if os.environ.get("NO_FIRST") == "1":   # 14B 500-target run in a fresh OUT: the smoke already did the hardware checks
        state["first"] = False

    def process(j, ok, msg):
        if ok:
            try:
                rec = run(j, state["first"])
                state["first"] = False
            except Exception as e:  # noqa: BLE001
                rec = dict(name=j["name"], group=j["group"], error=f"{e!r}"[:300])
        else:
            rec = dict(name=j["name"], group=j["group"], error=msg)
        rec["prep"] = msg
        with open(rp, "a") as f:
            f.write(json.dumps(rec) + "\n")
        log("done", j["name"], rec.get("seconds"), rec.get("error", ""))
        shutil.rmtree(f"data/{j['name']}", ignore_errors=True)

    deferred = []
    for j in todo:                       # pass 1: preparation overlaps the GPU work; killed preparations are deferred
        while j["name"] not in res:
            time.sleep(3)
        ok, msg = res[j["name"]]
        if not ok and ("rc -9" in msg or "exception" in msg):
            deferred.append(j)
            log("deferred", j["name"], msg)
        else:
            process(j, ok, msg)
        finished.add(j["name"])
    for k in list(MODELS):               # pass 2: deferred corpora prepared alone, GPU memory released meanwhile
        del MODELS[k]
    torch.cuda.empty_cache()
    for j in deferred:
        shutil.rmtree(f"data/{j['name']}", ignore_errors=True)
        try:
            ok, msg = prep(j)
        except Exception as e:  # noqa: BLE001
            ok, msg = False, f"prep exception {e!r}"[:300]
        log("prep solo", j["name"], ok, msg)
        process(j, ok, msg)
    log("ALL DONE")
    open(f"{OUT}/ALL_DONE", "w").write("1")


if __name__ == "__main__":
    main()
