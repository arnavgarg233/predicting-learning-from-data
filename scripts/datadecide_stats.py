"""Data-side statistics for the DataDecide pillar (docs/plan/datadecide_pillar_2026-10-12.md), one source at a time.

Sources: the 11 DataDecide eval sets (ev_*) and the 25 training recipes (rc_*). Every source is a uint16 token stream in
the DataDecide tokenizer (allenai/gpt-neox-olmo-dolma-v1_5, EOS 50279 between documents), written to data/datadecide/
samples/<slug>.npy. All outputs go to results/datadecide/stats/ and are written atomically, so any command can be rerun
after a disconnect and skips what is already done.

  python scripts/datadecide_stats.py list [--kind eval|recipe]           slugs, in priority order
  python scripts/datadecide_stats.py sample SLUG [SLUG ...]              eval: copy from --evaldata; recipe: HTTP range reads
  python scripts/datadecide_stats.py cpu SLUG                            -> stats/<slug>_cpu.json
  python scripts/datadecide_stats.py ref --model HF_ID SLUG [SLUG ...]   -> stats/<slug>_ref_<short>.json   (GPU)
  python scripts/datadecide_stats.py xent                                -> stats/xent.json (bigram cross-entropy matrix)
  python scripts/datadecide_stats.py merge                               -> stats/merged.json (adds gamma, power-law fits)

Recipe samples: N_RUNS contiguous runs of RUN_TOKENS tokens, each from a file drawn with probability proportional to its
token count and a uniform offset (what the learner's uniform instance sampling over the concatenated files sees), with
the partial documents at both run edges dropped. File lists: data/datadecide/recipe_paths.json.

Statistics per source (cpu):
  bytes_per_token        UTF-8 bytes of the decoded documents / tokens (EOS excluded)
  compression            lzma / bz2 / zlib bits per byte on the first 2^20 bytes of text (equal size for every source),
                         and lzma on up to 2^23 bytes
  ngram                  plug-in conditional entropies H_k = H(k-gram) - H((k-1)-gram), k = 1..4, bits/token, on the
                         first NGRAM_N tokens (equal size for every source)
  zipf                   exponent s of frequency ~ rank^-s over ranks 10..10^4
  doc_context            fraction of tokens with at least n tokens of in-document context, n in DOC_NS
  beta                   pair-correlation exponents (text/tok_stats.beta_stats, as for TinyStories and FineWeb) on the
                         first BETA_N tokens
Reference curves per source and model (ref; three byte-level BPE models as in text/refcurve/refcurve_fineweb.py):
  paired                 h(n) bits/byte for n in NS reference tokens, same targets for every n (refcurve_fineweb design)
  docpos                 bits/byte at in-document position n (document prefixes of DOCPOS_LEN reference tokens),
                         geometric bins up to DOCPOS_LEN - 1 tokens of context
"""
import argparse, bz2, hashlib, json, lzma, os, sys, time, unicodedata, urllib.request, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "text"))
sys.path.insert(0, str(ROOT / "text" / "refcurve"))
DD = ROOT / "data" / "datadecide"
SAMPLES = DD / "samples"
STATS = ROOT / "results" / "datadecide" / "stats"
EOS, V = 50279, 50280
EVALS = ["dolma_wiki", "dolma_stack", "dolma_reddit", "dolma_pes2o", "dolma_common-crawl", "dolma_books", "c4_en",
         "pile", "wikitext_103", "m2d2_s2orc", "ice"]  # in-distribution sets first
MODELS = os.environ.get("DD_MODELS", "HuggingFaceTB/SmolLM2-135M,EleutherAI/pythia-160m,openai-community/gpt2").split(",")
NS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]
DOC_NS = [1, 4, 16, 64, 256, 512, 1024, 2048, 4096]
E = os.environ.get
RUN_TOKENS = int(E("DD_RUN_TOKENS", 131072))
N_RUNS = int(E("DD_N_RUNS", 128))
NGRAM_N = int(E("DD_NGRAM_N", 200_000))
BETA_N = int(E("DD_BETA_N", 4_000_000))
NT = int(E("DD_NT", 3000))
DOCPOS_LEN = int(E("DD_DOCPOS_LEN", 1024))
DOCPOS_DOCS = int(E("DD_DOCPOS_DOCS", 800))
NOBYTE = "\x04\x06\x13\x14\x16\x1d"


def dump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=float))
    os.replace(tmp, path)


def recipes():
    return json.load(open(DD / "recipe_paths.json"))


def slugs(kind=None):
    rp = recipes()["recipes"]
    ev = [f"ev_{e}" for e in EVALS]
    rc = ["rc_" + v["hf_models"].split("DataDecide-")[1].replace("-<size>", "") for v in rp.values()]
    if E("DD_ONLY"):  # rehearsal: comma list of slugs to keep
        keep = E("DD_ONLY").split(",")
        ev, rc = [x for x in ev if x in keep], [x for x in rc if x in keep]
    return {"eval": ev, "recipe": rc, None: ev + rc}[kind]


def slug_name(slug):
    if slug.startswith("ev_"):
        return slug[3:]
    for r, v in recipes()["recipes"].items():
        if "rc_" + v["hf_models"].split("DataDecide-")[1].replace("-<size>", "") == slug:
            return r
    raise KeyError(slug)


# ------------------------------------------------------------------------------------------------------------- samples
def fetch(url, lo, n_bytes, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{lo + n_bytes - 1}"})
            with urllib.request.urlopen(req, timeout=120) as r:
                b = r.read()
            if len(b) == n_bytes:
                return b
        except Exception as ex:  # network hiccup: back off and retry
            print("retry", url, ex, flush=True)
        time.sleep(2 ** k)
    raise RuntimeError(f"range read failed: {url}")


def sample(slug, evaldata):
    out = SAMPLES / f"{slug}.npy"
    if out.exists():
        return
    SAMPLES.mkdir(parents=True, exist_ok=True)
    if slug.startswith("ev_"):
        a = np.fromfile(Path(evaldata) / f"{slug[3:]}.npy", dtype="<u2")
    else:
        rp = recipes()
        files = rp["recipes"][slug_name(slug)]["files"]
        ntok = np.array([s // 2 for _, s in files], float)
        rng = np.random.default_rng(int(hashlib.sha256(slug.encode()).hexdigest()[:8], 16))
        jobs = []
        for _ in range(N_RUNS):
            i = rng.choice(len(files), p=ntok / ntok.sum())
            off = int(rng.integers(0, max(int(ntok[i]) - RUN_TOKENS, 1)))
            jobs.append((rp["url_base"] + files[i][0], 2 * off, 2 * min(RUN_TOKENS, int(ntok[i]))))
        with ThreadPoolExecutor(8) as ex:
            runs = list(ex.map(lambda j: np.frombuffer(fetch(*j), dtype="<u2"), jobs))
        parts = []
        for r in runs:
            cut = np.flatnonzero(r == EOS)
            if len(cut) >= 2:
                parts.append(r[cut[0] + 1:cut[-1] + 1])  # whole documents, each followed by its EOS
        a = np.concatenate(parts)
    assert a.max() < V
    tmp = out.with_suffix(".tmp")
    a.astype("<u2").tofile(tmp)
    os.replace(tmp, out)
    print(slug, len(a), "tokens", flush=True)


def load(slug):
    return np.fromfile(SAMPLES / f"{slug}.npy", dtype="<u2")


def docs_of(a):
    docs = (d[d != EOS] for d in np.split(a, np.flatnonzero(a == EOS) + 1))
    return [d for d in docs if len(d)]


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(DD / "tokenizer.json"))


def texts(a, tok):
    return tok.decode_batch([d.tolist() for d in docs_of(a)], skip_special_tokens=False)


# ------------------------------------------------------------------------------------------------------------- cpu
def ngram_entropies(t, kmax=4):
    H, prev = [], 0.0
    for k in range(1, kmax + 1):
        rows = np.lib.stride_tricks.sliding_window_view(t.astype(np.int64), k)
        key = rows[:, 0].copy()
        for j in range(1, k):
            key = key * V + rows[:, j] if k <= 3 else hash_mix(key, rows[:, j])
        _, c = np.unique(key, return_counts=True)
        p = c / c.sum()
        Hk = float(-(p * np.log2(p)).sum())
        H.append(Hk - prev)
        prev = Hk
    return H


def hash_mix(key, col):  # 64-bit mixing for k >= 4 (V^4 overflows int64); collisions negligible at these sizes
    return (key * np.int64(1_000_003)) ^ (col + np.int64(0x9E3779B97F4A7C15 & 0x7FFFFFFFFFFFFFFF))


def cpu(slug):
    out = STATS / f"{slug}_cpu.json"
    if out.exists():
        return
    from tok_stats import beta_stats
    a = load(slug)
    tok = tokenizer()
    txt = texts(a, tok)
    body = a[a != EOS]
    blob = "\n".join(txt).encode("utf-8")
    nb = sum(len(s.encode("utf-8")) for s in txt)
    one = blob[:2 ** 20]
    comp = {f"{n}_bpb_1MB": 8 * len(f(one)) / len(one)
            for n, f in (("lzma", lambda b: lzma.compress(b, preset=9)), ("bz2", lambda b: bz2.compress(b, 9)),
                         ("zlib", lambda b: zlib.compress(b, 9)))}
    big = blob[:2 ** 23]
    comp["lzma_bpb_8MB"] = 8 * len(lzma.compress(big, preset=9)) / len(big)
    comp["n_bytes_8MB"] = len(big)
    cnt = np.bincount(body, minlength=V).astype(float)
    p = cnt[cnt > 0] / cnt.sum()
    f = np.sort(cnt[cnt > 0])[::-1]
    rk = np.arange(1, len(f) + 1)
    sel = (rk >= 10) & (rk <= min(10_000, len(f)))
    lens = np.array([len(d) for d in docs_of(a)])
    doc_ctx = {n: float(np.clip(lens - n, 0, None).sum() / lens.sum()) for n in DOC_NS}
    t0 = time.time()
    bt = a[:BETA_N].astype(np.int64)
    beta = beta_stats(bt, V, nb * len(bt) / max(len(body), 1))
    dump(dict(slug=slug, name=slug_name(slug), n_tokens=int(len(body)), n_docs=int(len(lens)), n_bytes=int(nb),
              bytes_per_token=nb / len(body), compression=comp, unigram_bits=float(-(p * np.log2(p)).sum()),
              ngram_n=int(min(NGRAM_N, len(body))), H_cond=ngram_entropies(body[:NGRAM_N]),
              zipf_s=float(-np.polyfit(np.log(rk[sel]), np.log(f[sel]), 1)[0]),
              doc_len_tokens=dict(mean=float(lens.mean()), median=float(np.median(lens))), doc_context=doc_ctx,
              beta=beta, beta_seconds=time.time() - t0), out)
    print(slug, "cpu done", flush=True)


# ------------------------------------------------------------------------------------------------------------- ref
def ref(model, slug_list, device):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from refcurve_fineweb import bytes_to_unicode, summarize
    short = model.split("/")[-1]
    todo = [s for s in slug_list if not (STATS / f"{s}_ref_{short}.json").exists()]
    if not todo:
        return
    dev = torch.device(device if device != "cuda" or torch.cuda.is_available() else "cpu")
    rt = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, **({"attn_implementation": E("DD_ATTN")} if E("DD_ATTN") else {})).to(dev).eval()
    u2b = {v: k for k, v in bytes_to_unicode().items()}
    Vm = max(len(rt), m.get_output_embeddings().weight.shape[0])
    tb = np.zeros(Vm, np.int32)
    for t, i in rt.get_vocab().items():
        # byte-level BPE: one character per byte; added tokens (e.g. GPT-NeoX's runs of spaces) are stored as plain text
        tb[i] = len(t) if all(c in u2b for c in t) else len(t.encode("utf-8"))
    nmax = max(NS)
    dtok = tokenizer()
    for slug in todo:
        t0 = time.time()
        rng = np.random.default_rng(0)
        txt = texts(load(slug), dtok)
        order = rng.permutation(len(txt))
        ids, kept, tot, nskip = [], [], 0, 0
        for j in order:  # enough documents for NT paired targets and DOCPOS_DOCS long prefixes
            if not unicodedata.is_normalized("NFC", txt[j]):  # Pythia's tokenizer applies NFC; same documents for every model
                continue
            if any(c in txt[j] for c in NOBYTE):  # SmolLM2's vocab has no token for these control bytes and drops them; same documents for every model
                nskip += 1
                continue
            x = rt(txt[j], add_special_tokens=False)["input_ids"]
            ids.append(np.asarray(x, np.int64))  # arrays, not lists: a quarter of the memory for the document pool
            kept.append(j)
            tot += max(len(x) - nmax, 0)
            if tot >= 20 * NT and sum(len(y) >= DOCPOS_LEN for y in ids) >= DOCPOS_DOCS:
                break
        print(f"{slug} {short}: skipped {nskip} documents with control bytes missing from SmolLM2's vocabulary", flush=True)
        for x, j in zip(ids, kept):
            assert int(tb[x].sum()) == len(txt[j].encode("utf-8")), f"{model}: token byte lengths do not add up"
        # paired curve (refcurve_fineweb design)
        ok = np.array([max(len(x) - nmax, 0) for x in ids])
        nt = int(min(NT, ok.sum()))
        res = dict(model=model, slug=slug, n_docs_read=len(ids))
        if nt >= 100:
            flat = np.sort(rng.choice(int(ok.sum()), nt, replace=False))
            cs = np.cumsum(ok)
            doc = np.searchsorted(cs, flat, side="right")
            pos = flat - (cs[doc] - ok[doc]) + nmax
            tgt = np.array([ids[d][p] for d, p in zip(doc, pos)])
            ctxb = np.array([[tb[ids[d][p - n:p]].sum() for n in NS] for d, p in zip(doc, pos)])
            loss = np.zeros((nt, len(NS)), np.float32)
            for c in range(0, nt, 250):
                for k, n in enumerate(NS):
                    # DD_REF_TOKENS caps tokens per forward (memory on MPS/CPU); unset = 250 targets per batch as before
                    step = max(4, int(E("DD_REF_TOKENS")) // n) if E("DD_REF_TOKENS") else 250
                    for c2 in range(c, min(c + 250, nt), step):
                        sl = slice(c2, min(c2 + step, c + 250, nt))
                        x = torch.as_tensor(np.stack([ids[d][p - n:p] for d, p in zip(doc[sl], pos[sl])]), device=dev)
                        with torch.no_grad():
                            h = m.base_model(x).last_hidden_state[:, -1]
                            lp = torch.log_softmax(m.get_output_embeddings()(h).float(), -1)
                        loss[sl, k] = -lp[torch.arange(len(x), device=dev), torch.as_tensor(tgt[sl], device=dev)].cpu().numpy()
                        if dev.type == "mps":
                            torch.mps.empty_cache()
            res["paired"] = summarize(short, model, loss / np.log(2), doc, tb[tgt].astype(float), ctxb, NS, len(ids))
        else:
            res["paired"] = None
        res["paired_note"] = f"{nt} targets with >= {nmax} in-document predecessors"
        # in-document position curve on document prefixes
        long = [x[:DOCPOS_LEN] for x in ids if len(x) >= DOCPOS_LEN][:DOCPOS_DOCS]
        if len(long) >= 20 and not E("DD_NO_DOCPOS"):  # DD_NO_DOCPOS: same document pool, docpos curve skipped
            X = torch.as_tensor(np.stack(long), device=dev)
            L = np.zeros((len(long), DOCPOS_LEN - 1), np.float32)
            for c in range(0, len(long), 8):
                with torch.no_grad():
                    lp = torch.log_softmax(m(X[c:c + 8]).logits[:, :-1].float(), -1)
                    L[c:c + 8] = -lp.gather(-1, X[c:c + 8, 1:, None])[..., 0].cpu().numpy()
            B = tb[np.array(long)[:, 1:]].astype(float)
            CB = np.cumsum(tb[np.array(long)], 1)[:, :-1].astype(float)
            edges = np.unique(np.round(np.geomspace(1, DOCPOS_LEN - 1, 25)).astype(int))
            rows = []
            for lo, hi in zip(edges[:-1], edges[1:]):  # positions with n in [lo, hi) tokens of context
                sl = slice(lo - 1, hi - 1)
                rows.append(dict(n_lo=int(lo), n_hi=int(hi), bits_per_byte=float(L[:, sl].sum() / np.log(2) / B[:, sl].sum()),
                                 context_bytes_exact_mean=float(CB[:, sl].mean())))
            res["docpos"] = dict(n_docs=len(long), len=DOCPOS_LEN, rows=rows)
        else:
            res["docpos"] = None
        res["seconds"] = time.time() - t0
        dump(res, STATS / f"{slug}_ref_{short}.json")
        print(slug, short, f"{res['seconds']:.0f}s", flush=True)
    del m
    if dev.type == "cuda":
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------------------------------------- xent
class Bigram:
    """Interpolated absolute-discount bigram model (d = 0.75) over the DataDecide vocabulary, EOS pairs excluded."""

    def __init__(self, t, d=0.75):
        a, b = t[:-1].astype(np.int64), t[1:].astype(np.int64)
        k = (a != EOS) & (b != EOS)
        self.keys, self.c = np.unique(a[k] * V + b[k], return_counts=True)
        first = self.keys // V
        self.ca = np.bincount(first, weights=self.c, minlength=V)
        self.n1 = np.bincount(first, minlength=V).astype(float)
        u = np.bincount(t[t != EOS], minlength=V).astype(float) + 0.5
        self.pu, self.d = u / u.sum(), d

    def bits(self, t):
        a, b = t[:-1].astype(np.int64), t[1:].astype(np.int64)
        k = (a != EOS) & (b != EOS)
        a, b = a[k], b[k]
        key = a * V + b
        i = np.clip(np.searchsorted(self.keys, key), 0, len(self.keys) - 1)
        cab = np.where(self.keys[i] == key, self.c[i], 0.0)
        ca = self.ca[a]
        seen = ca > 0
        p = np.where(seen, np.maximum(cab - self.d, 0) / np.maximum(ca, 1) + self.d * self.n1[a] / np.maximum(ca, 1) * self.pu[b],
                     self.pu[b])
        return float(-np.log2(p).mean()), int(len(a))


def xent():
    out = STATS / "xent.json"
    have = [s for s in slugs() if (SAMPLES / f"{s}.npy").exists()]
    prev = json.loads(out.read_text()) if out.exists() else {}
    if prev.get("sources") == have:
        return
    data = {s: load(s) for s in have}
    split = {s: int(0.9 * len(a)) for s, a in data.items()}
    M = {}
    for s in have:  # model trained on the first 90% of s, scored on every other source and on the last 10% of s
        bg = Bigram(data[s][:split[s]])
        M[s] = {t: bg.bits(data[t][split[t]:] if t == s else data[t])[0] for t in have}
        print("xent", s, flush=True)
    dump(dict(sources=have, unit="bits per token (bigram, abs. discount 0.75)", model_on_first_90pct=True, xent=M), out)


# ------------------------------------------------------------------------------------------------------------- merge
GAMMA_WINDOWS = [(2, 32), (4, 64), (8, 128), (32, 512)]


def powerlaw_fit(nb, h):
    """h(n) = h_inf + a n^-g over context bytes nb (least squares in h); None if it does not converge."""
    from scipy.optimize import curve_fit
    try:
        (hi, a, g), _ = curve_fit(lambda x, hi, a, g: hi + a * x ** -g, nb, h, p0=(h[-1] * 0.9, h[0] - h[-1], 0.5),
                                  maxfev=20000, bounds=([0, 0, 0.01], [h.min(), np.inf, 5]))
        return dict(h_inf=float(hi), a=float(a), g=float(g))
    except Exception:
        return None


def merge():
    from tok_stats import gamma_stats
    out = {}
    for s in slugs():
        f = STATS / f"{s}_cpu.json"
        rec = dict(kind="eval" if s.startswith("ev_") else "recipe", name=slug_name(s))
        if f.exists():
            rec.update(json.loads(f.read_text()))
        rec["ref"] = {}
        for mname in MODELS:
            short = mname.split("/")[-1]
            g = STATS / f"{s}_ref_{short}.json"
            if not g.exists():
                continue
            r = json.loads(g.read_text())
            e = dict(paired=r["paired"], docpos=r["docpos"])
            if r["paired"] and "bytes_per_token" in rec:
                rows = r["paired"]["rows"]
                e["gamma"] = {f"{w[0]}-{w[1]}": gamma_stats({short: r["paired"]}, rec["bytes_per_token"],
                                                             rec["beta"]["beta"]["tok_1_200"], w)[short]
                              for w in GAMMA_WINDOWS if w[1] * rec["bytes_per_token"] <= rows[-1]["context_bytes_exact_mean"]}
                nb = np.array([x["context_bytes_exact_mean"] for x in rows])
                h = np.array([x["bits_per_byte"] for x in rows])
                e["fit_paired"] = powerlaw_fit(nb[nb >= 8], h[nb >= 8])
            if r["docpos"]:
                rows = r["docpos"]["rows"]
                nb = np.array([x["context_bytes_exact_mean"] for x in rows])
                h = np.array([x["bits_per_byte"] for x in rows])
                e["fit_docpos"] = powerlaw_fit(nb[nb >= 8], h[nb >= 8])
            rec["ref"][short] = e
        out[s] = rec
    xf = STATS / "xent.json"
    dump(dict(sources=out, xent=json.loads(xf.read_text()) if xf.exists() else None, models=MODELS), STATS / "merged.json")
    print("merged", sum(bool(v["ref"]) for v in out.values()), "sources with reference curves", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["list", "sample", "cpu", "ref", "xent", "merge"])
    ap.add_argument("slugs", nargs="*")
    ap.add_argument("--kind", default=None)
    ap.add_argument("--model", default=MODELS[0])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--evaldata", default=str(DD / "evaldata"))
    a = ap.parse_intermixed_args()
    if a.cmd == "list":
        print("\n".join(slugs(a.kind)))
    elif a.cmd == "sample":
        for s in a.slugs:
            sample(s, a.evaldata)
    elif a.cmd == "cpu":
        for s in a.slugs:
            cpu(s)
    elif a.cmd == "ref":
        ref(a.model, a.slugs, a.device)
    elif a.cmd == "xent":
        xent()
    else:
        merge()


if __name__ == "__main__":
    main()
