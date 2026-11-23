"""Refit of published compute-scaling minima (Limisiewicz et al., "Compute Optimal Tokenization", Tables 2 and 10).

    python scripts/limisiewicz_refit.py [--out PATH]   # -> results/external/limisiewicz2026_refit.json

Input: results/external/limisiewicz2026_table10.json, the lowest C4-validation bits per byte at each compute budget
(1e19-2e21 FLOPs) as printed in the paper, for subword tokenizers (Table 2) and for byte-latent models (Table 10).
Exponent of (BPB - E) against compute, per tokenizer: by log-log slope at fixed floors E, and by least squares on log BPB
with one floor shared by all tokenizers of a family and one or separate exponents. The text quotes the character /
Llama-3 BPE exponent ratio at the jointly fitted floor (per-tokenizer exponents) and its range over floors 0.3-0.75.
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IN = os.path.join(ROOT, "results/external/limisiewicz2026_table10.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "results/external/limisiewicz2026_refit.json"))
    a = ap.parse_args()
    d = json.load(open(IN))
    lc = np.log(np.array(d["C"]) / 1e20)
    out = dict(source=os.path.relpath(IN, ROOT), families={})
    for name, TT in (("latent (BLT)", d["T"]), ("subword", d["table2_subword"]["T"])):
        Ts = list(TT)
        Y = {t: np.array(v) for t, v in TT.items()}
        fam = dict(fixed_floor={str(E): {t: float(-np.polyfit(lc, np.log(Y[t] - E), 1)[0]) for t in Ts} for E in (0.5, 0.6, 0.7)})

        def res(p, shared):
            E = p[0]
            o = []
            for i, t in enumerate(Ts):
                al = p[1] if shared else p[1 + i]
                A = np.exp(p[1 + (1 if shared else len(Ts)) + i])
                o.append(np.log(E + A * np.exp(-al * lc)) - np.log(Y[t]))
            return np.concatenate(o)
        for shared in (True, False):
            na = 1 if shared else len(Ts)
            r = least_squares(lambda p: res(p, shared), [0.6] + [0.2] * na + [np.log(0.35)] * len(Ts),
                              bounds=([0] + [0] * na + [-10] * len(Ts), [0.84] + [3] * na + [5] * len(Ts)))
            fam["shared_exponent" if shared else "per_tokenizer_exponent"] = dict(
                E=float(r.x[0]), exponents=dict(zip(["all"] if shared else Ts, map(float, r.x[1:1 + na]))),
                rms_log=float(np.sqrt(np.mean(r.fun ** 2))))
        if name == "subword":
            al = lambda t, E: -np.polyfit(lc, np.log(Y[t] - E), 1)[0]  # noqa: E731
            fam["char_over_llama3_by_floor"] = {str(E): float(al("char_1.01", E) / al("bpe_llama3_4.57", E))
                                                for E in (0.3, 0.5, 0.6, 0.7, 0.75, 0.8)}
            pe = fam["per_tokenizer_exponent"]["exponents"]
            fam["char_over_llama3_joint_floor"] = pe["char_1.01"] / pe["bpe_llama3_4.57"]
        out["families"][name] = fam
    json.dump(out, open(a.out, "w"), indent=1)
    s = out["families"]["subword"]
    print("subword joint floor E %.3f; char/Llama3 exponent ratio %.2f; over floors 0.3-0.75: %s" % (
        s["per_tokenizer_exponent"]["E"], s["char_over_llama3_joint_floor"],
        {k: round(v, 2) for k, v in s["char_over_llama3_by_floor"].items()}))
    for n, f in out["families"].items():
        for k in ("shared_exponent", "per_tokenizer_exponent"):
            print(n, k, "E %.3f" % f[k]["E"], {t: round(v, 3) for t, v in f[k]["exponents"].items()}, "rms_log %.4f" % f[k]["rms_log"])


if __name__ == "__main__":
    main()
