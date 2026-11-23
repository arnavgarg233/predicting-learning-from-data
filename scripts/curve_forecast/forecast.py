"""Phase 4: information inputs and the zero-training forecast A0 for the test corpora, before any learner training.

    python scripts/curve_forecast/forecast.py      # -> results/curve_forecast/forecast.json (sha256 printed)

Reads only results/curve_forecast/locked.json, results/curve_forecast/corpora.json (the test list), each corpus's training
prefix and ref_stats partition (data/cf/<c>/tok) and its reference curve (results/curve_forecast/<c>/refcurve, the primary
model only). Computes the best n-gram cross-entropy at 2^23 (forecast_multicorpus.ng_curve, as for the development corpora),
the shape parameters of the locked shape, the shape at every budget, and A0 at 2^23 and 2^25 with the locked level map.
No learner run of a test corpus exists when this runs.
"""
import datetime
import hashlib
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import core as C  # noqa: E402

ROOT = C.ROOT
RES = os.path.join(ROOT, "results/curve_forecast")
BUDGETS = (19, 20, 23, 25)
TARGETS = (23, 25)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def main():
    sys.path.insert(0, C.SCRIPTS)
    import forecast_multicorpus as FM
    lk = json.load(open(os.path.join(RES, "locked.json")))
    ch, K = lk["chosen"], lk["constants"]
    cl = json.load(open(os.path.join(RES, "corpora.json")))
    out = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
               locked_sha256=sha(os.path.join(RES, "locked.json")), chosen=ch,
               code_sha256={os.path.relpath(p, ROOT): sha(p) for p in
                            [os.path.join(HERE, n) for n in sorted(os.listdir(HERE)) if n.endswith(".py")] +
                            [os.path.join(C.SCRIPTS, n) for n in ("forecast_neff.py", "forecast_input_variants.py", "forecast_multicorpus.py", "ngram_curve_ts.py")]},
               corpora={})
    for c in cl["test"]:
        name, model = c["name"], c["ref_model"]
        d = os.path.join(ROOT, "data/cf", name, "tok")
        rp = os.path.join(RES, name, "refcurve", "refcurve_fineweb.json")
        hr = C.FN.HRef(rp, model)
        st = C.ref_stats(hr)
        FM.ng_curve(name, d, 23, RES)
        st["ngram23"] = C.best_ngram(os.path.join(RES, name, "ngram_23.json"))
        nu, n0 = C.shape_params(ch["shape"], st, K)
        level, coef = ch["level"], K["level_coef"]
        if not all(np.isfinite(C.level_X(level, st))):
            level, coef = "ref", K["level_coef_ref_fallback"]
        out["corpora"][name] = dict(
            family=c["family"], ref_model=model, ref_stats=st, shape=dict(nu=nu, n0=n0),
            S={str(k): C.S(hr, nu, n0, k) for k in BUDGETS}, level_used=level,
            A0={str(k): C.A0(hr, st, nu, n0, level, ch["level_mode"], coef, k) for k in TARGETS},
            refcurve_sha256=hr.sha, ref_n_targets=json.load(open(rp))[model].get("n_targets"),
            inputs={os.path.relpath(p, ROOT): sha(p) for p in (os.path.join(d, "tok_bytes_train.npy"), os.path.join(d, "tok_bytes_ref.npy"))})
        print(name, model, {k: round(v, 3) for k, v in out["corpora"][name]["A0"].items()}, flush=True)
    p = os.path.join(RES, "forecast.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print("sha256", sha(p), os.path.relpath(p, ROOT))


if __name__ == "__main__":
    main()
