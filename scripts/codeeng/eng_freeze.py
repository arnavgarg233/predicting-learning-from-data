"""Test 2 (English forward test): frame, seeded draw, exclusions, outlier rule and the frozen English line, written once before
any English corpus is prepared (docs/analysis/code25m_english.md, Test 2).

    python scripts/codeeng/eng_freeze.py   # -> results/fwd_english/fall_line.json, candidates_e1.json, candidates_e2.json

Frames (third-party lists, listings saved to ops/state/codeeng/eng_frames/ on 2026-11-01):
  1. Common Pile v0.1 (Kandpal et al. 2025): the unfiltered source datasets of the common-pile organisation on Hugging Face
     (not *_filtered, not the mixtures comma_v0.1_training_dataset / raw_v0.1_parquet). Size: datasets-server parquet bytes.
  2. Pile of Law (Henderson et al. 2022): the subsets of pile-of-law/pile-of-law (data/train.<subset>*.jsonl.xz). Size: xz
     bytes summed over a subset's files.
Floor: at least 2^26 bytes of parquet (Common Pile) or 2^23 bytes of xz (Pile of Law, about 2^26 bytes of text at the 6-8x
xz ratio of English prose). A drawn corpus whose preparation fails (too little text) is replaced by its pod's next spare.
Exclusions (as scripts/extbench/draw.py: everything whose whole documents can occur in a corpus the paper uses, code, or not
English): see EXCL below.
Draw: seed 20261101. Each frame's eligible names are sorted and permuted with numpy default_rng(seed + j), j = 0 Common Pile,
1 Pile of Law; Common Pile first, Pile of Law after it as top-up (the precedent of draw.py: Pile first, Dolma top-up). The first
30 are drawn, the rest are spares; drawn and spares are dealt alternately to pods e1 and e2.
Calibration (frozen now): the 29 English corpora with own falls (results/reanalysis/fall_diagnosis.json rows, kind
english, 3.3M learner). Outlier rule for pathological curves: a corpus is set aside if its measured fall or its own fall lies
more than 4 scaled MADs (1.4826 x MAD) from the calibration median of that quantity. The calibration medians and MADs are
frozen here and applied unchanged to the new corpora. Frozen line: OLS of fall on own fall over the clean calibration corpora;
frozen kind mean: their mean fall. Context arm (also frozen): the line with the fall rule only.
"""
import datetime
import json
import os
import subprocess

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "results", "fwd_english")
FR = os.path.join(ROOT, "ops", "state", "codeeng", "eng_frames")
SEED = 20261101
N = 30
EXCL = {
    "cp_arxiv_abstracts": "arxiv used", "cp_arxiv_papers": "arxiv used",
    "cp_biodiversity_heritage_library": "public-domain books that can also be in Project Gutenberg (pg19)",
    "cp_caselaw_access_project": "caselaw used", "cp_cccc": "Common Crawl pages (C4, FineWeb-Edu, CC-News are CC extractions)",
    "cp_data_provenance_initiative": "instruction data built from datasets the paper uses (as Flan in draw.py)",
    "cp_doab": None, "cp_foodista": None, "cp_github_archive": "code (GitHub issues and pull requests)",
    "cp_library_of_congress": "public-domain books that can also be in Project Gutenberg (pg19)",
    "cp_news": "news sites crawled by CC-News (cc_news used)", "cp_oercommons": None,
    "cp_peS2o": "Semantic Scholar papers (sci_abstracts are S2 abstracts)",
    "cp_pre_1929_books": "public-domain books that can also be in Project Gutenberg (pg19)",
    "cp_project_gutenberg": "pg19 used", "cp_public_domain_review": None, "cp_pubmed": "pubmed used",
    "cp_python_enhancement_proposals": "code documentation", "cp_stackexchange": "stackexchange used", "cp_stackv2": "code",
    "cp_ubuntu_irc": "pile_ubuntu_irc used", "cp_usgpo": "US Government Publishing Office: bills (billsum) and reports (govreport)",
    "cp_uspto": "US patents (patents used)", "cp_wikimedia": "Wikipedia",
    "pol_courtlisteneropinions": "court opinions (caselaw, scotus used)",
    "pol_courtlistenerdocketentries": "court documents that include opinions (caselaw, scotus used)",
    "pol_scotus_docket_entries": "Supreme Court documents that include opinions (scotus used)",
    "pol_us_bills": "US bills (billsum used)", "pol_echr": "ECHR judgments (ecthr used)",
    "pol_r_legaldvice": "Reddit (dolma_reddit used)", "pol_euro_parl": "multilingual, not English",
    "pol_canadian_decisions": "English and French", "pol_cc_casebooks": "casebooks reproduce court opinions (caselaw, scotus used)",
}


def get(url, path):
    if not os.path.exists(path):
        r = subprocess.run(["curl", "-s", "-m", "60", url], capture_output=True, text=True, check=True)
        open(path, "w").write(r.stdout)
    return json.load(open(path))


def frames():
    os.makedirs(FR, exist_ok=True)
    org = get("https://huggingface.co/api/datasets?author=common-pile&limit=500", os.path.join(FR, "common_pile_org.json"))
    cp = {}
    for d in sorted(x["id"].split("/")[1] for x in org):
        if d.endswith("_filtered") or d in ("comma_v0.1_training_dataset", "raw_v0.1_parquet") or d.startswith("stackv2_"):
            continue
        s = get(f"https://datasets-server.huggingface.co/size?dataset=common-pile/{d}", os.path.join(FR, f"cp_size_{d}.json"))
        cp["cp_" + d] = (s.get("size") or {}).get("dataset", {}).get("num_bytes_parquet_files")
    tree = get("https://huggingface.co/api/datasets/pile-of-law/pile-of-law/tree/main/data", os.path.join(FR, "pile_of_law_tree.json"))
    pol = {}
    for f in tree:
        p = f["path"].split("/")[-1]
        if p.startswith("train."):
            k = "pol_" + p[len("train."):].split(".")[0]
            pol[k] = pol.get(k, 0) + f.get("size", 0)
    return cp, pol


def eligible(sizes, floor):
    el, ex = [], {}
    for k, s in sorted(sizes.items()):
        if EXCL.get(k):
            ex[k] = EXCL[k]
        elif not s or s < floor:
            ex[k] = f"below the size floor ({s} bytes)"
        else:
            el.append(k)
    return el, ex


def mad_keep(v, med, mad, k=4.0):
    return np.abs(np.asarray(v, float) - med) <= k * mad


def main():
    if os.path.exists(os.path.join(OUT, "fall_line.json")):
        raise SystemExit("fall_line.json exists; not overwriting")
    import sys
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    import prep_corpus as P
    cp, pol = frames()
    e1, x1 = eligible(cp, 2 ** 26)
    e2, x2 = eligible(pol, 2 ** 23)
    assert set(e1) == {f"cp_{x}" for x in P.ENG2_CP} and set(e2) == {f"pol_{x}" for x in P.ENG2_POL}, (e1, e2)
    o1 = [e1[i] for i in np.random.default_rng(SEED).permutation(len(e1))]
    o2 = [e2[i] for i in np.random.default_rng(SEED + 1).permutation(len(e2))]
    order = o1 + o2
    drawn, spares = order[:N], order[N:]
    mk = lambda n, pos: dict(name=n, family="english", ref_model="SmolLM2-135M", position=pos)
    pods = {}
    for p in (1, 2):
        lst = [mk(n, i + 1) for i, n in enumerate(drawn) if i % 2 == p - 1]
        sp = [mk(n, N + i + 1) for i, n in enumerate(spares) if i % 2 == p - 1]
        pods[f"e{p}"] = dict(pod=f"e{p}", list=lst, pools=dict(english=sp))
    # calibration
    D = json.load(open(os.path.join(ROOT, "results", "reanalysis", "fall_diagnosis.json")))
    rows = [r for r in D["rows"] if r["kind"] == "english"]
    y = np.array([r["fall"] for r in rows]); x = np.array([r["x"] for r in rows])
    med = dict(fall=float(np.median(y)), x=float(np.median(x)))
    mad = dict(fall=float(1.4826 * np.median(np.abs(y - med["fall"]))), x=float(1.4826 * np.median(np.abs(x - med["x"]))))
    kf = mad_keep(y, med["fall"], mad["fall"]); kx = mad_keep(x, med["x"], mad["x"])
    keep = kf & kx
    b, a = np.polyfit(x[keep], y[keep], 1)
    bf, af = np.polyfit(x[kf], y[kf], 1)
    now = subprocess.run(["date"], capture_output=True, text=True).stdout.strip()
    out = dict(date=now, design="docs/analysis/code25m_english.md (Test 2)", rule=__doc__, seed=SEED,
               frames=dict(common_pile=dict(sizes_parquet_bytes=cp, eligible=e1, excluded=x1, seeded_order=o1),
                           pile_of_law=dict(sizes_xz_bytes=pol, eligible=e2, excluded=x2, seeded_order=o2)),
               order=order, drawn=drawn, spares=spares, pods={k: [c["name"] for c in v["list"]] for k, v in pods.items()},
               calibration=dict(n=len(rows), corpora=[r["corpus"] for r in rows], median=med, scaled_mad=mad, k_mad=4.0,
                                set_aside=[r["corpus"] for r, k in zip(rows, keep) if not k],
                                set_aside_fall_rule=[r["corpus"] for r, k in zip(rows, kf) if not k],
                                n_clean=int(keep.sum()), line=dict(a=float(a), b=float(b)), kind_mean=float(y[keep].mean()),
                                context_fall_rule_only=dict(n=int(kf.sum()), line=dict(a=float(af), b=float(bf)),
                                                            kind_mean=float(y[kf].mean()))),
               scoring="scripts/codeeng/score_t2.py; primary: r.m.s. fall error of the frozen line against the frozen kind mean "
                       "on the new corpora that pass the outlier rule (frozen medians and MADs), paired corpus bootstrap 20,000 "
                       "draws, default_rng(0), 95% interval; REAL positive if below 1, REAL negative if above 1, tie otherwise")
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "fall_line.json"), "w"), indent=1)
    for k, v in pods.items():
        json.dump(v, open(os.path.join(OUT, f"candidates_{k}.json"), "w"), indent=1)
    print(now); print("order", order); print("drawn", len(drawn), "spares", spares)
    print("calibration", out["calibration"]["set_aside"], "line", out["calibration"]["line"], "mean", out["calibration"]["kind_mean"],
          "fall-rule-only", out["calibration"]["context_fall_rule_only"])


if __name__ == "__main__":
    main()
