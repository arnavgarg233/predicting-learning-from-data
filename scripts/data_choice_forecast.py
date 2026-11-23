"""Forecast the 1B-scale held-out loss of models trained on public pretraining corpora, from data statistics alone.

    python scripts/data_choice_forecast.py        -> results/data_choice/forecast.json

The predictor is task E of scripts/datadecide_predict.py, applied unchanged: an affine map of one or two statistics,
the statistics chosen by leave-one-family-out MAE over the six DataDecide corpus families, then fitted on all 25
DataDecide recipes. Target: bits/byte at 1B (100 tokens per parameter), averaged with equal weight over the 11
DataDecide eval sets (all11) or over the 4 sets no recipe trains on (ood4). The new corpora enter only through their
statistics (scripts/data_choice_stats.py); nothing about them is used to choose or fit anything.

Variants (feature pools): ref_lm (all statistics, the 25-recipe row with pairwise accuracy 0.75), noref (no reference
LM), mismatch (the theory term alone), and one pool per reference LM (non-reference statistics plus that model's
statistics only), to expose reference-LM bias.

Intervals: 1000 bootstrap resamples of the 25 training recipes (selection and fit repeated in each). Also reported:
forecasts with the corpus's own DataDecide family removed from training (for corpora that are DataDecide recipes),
and the unigram cross-entropy rival (rivals.unigram_ce).
"""
import hashlib, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import datadecide_predict as ddp  # noqa: E402

DD = ROOT / "results" / "datadecide"
DC = ROOT / "results" / "data_choice"
NBOOT = int(os.environ.get("DC_NBOOT", 1000))
LABEL = {"fineweb": "FineWeb", "fineweb_edu": "FineWeb-Edu", "c4": "C4", "dclm_baseline": "DCLM-baseline",
         "refinedweb": "RefinedWeb", "pile": "The Pile (uncopyrighted)", "dolma": "Dolma v1.7",
         "slimpajama": "SlimPajama", "redpajama_1t": "RedPajama-1T (sample)"}
# corpora that are themselves DataDecide recipes, and the DataDecide families that share their text
OVERLAP = {"c4": dict(recipe="C4", families=["c4"]),
           "dclm_baseline": dict(recipe="DCLM-Baseline", families=["dclm", "dclm_dolma_mix"]),
           "dolma": dict(recipe="Dolma1.7", families=["dolma", "dclm_dolma_mix"]),
           "fineweb_edu": dict(recipe="FineWeb-Edu", families=["fineweb"]),
           "refinedweb": dict(recipe="Falcon", families=["falcon"])}
RELATED = {"fineweb": dict(note="parent corpus of DataDecide's FineWeb-Edu and FineWeb-Pro", families=["fineweb"])}


ROW_LABELS = {
    "mismatch": "PRIMARY. ours_mismatch of datadecide_predict.py: affine in the eval-set mismatch M (bigram cross-entropy "
                "of the DataDecide eval text under the corpus sample minus the eval text's own), fitted on the 25 recipes. "
                "No reference LM.",
    "ref_gpt2-2019_only": "reference-LM row with GPT-2 only (WebText, 2019; predates every corpus ranked here)",
    "ref_gpt2_only": "reference-LM row with GPT-2 only (WebText, 2019; predates every corpus ranked here)",
    "ref_SmolLM2-135M_only": "SECONDARY, possibly contaminated: SmolLM2-135M was pretrained on FineWeb-Edu and DCLM",
    "ref_pythia-160m_only": "not used for ranking: Pythia was trained on the Pile, which is ranked here",
    "ref_lm": "possibly contaminated: pool includes SmolLM2 (FineWeb-Edu, DCLM) and Pythia (Pile) statistics",
    "noref": "all non-reference statistics (selection as ours_select_noref)",
}


def sets(pool, col):
    return [(col[a],) for a in pool] + [(col[a], col[b]) for i, a in enumerate(pool) for b in pool[i + 1:]]


def inner_mae(F, y, fam, cands, tr):
    """Leave-one-family-out MAE inside tr of every candidate set, as ddp.lofo(F, y, fam, [c], tr) computes it, with
    the least-squares fits batched over candidates of equal size (pseudo-inverse = lstsq's minimum-norm solution)."""
    err = np.zeros(len(cands))
    groups = {}
    for i, c in enumerate(cands):
        groups.setdefault(len(c), []).append(i)
    for k, idx in groups.items():
        C = np.array([cands[i] for i in idx], int).reshape(len(idx), k)
        for f in np.unique(fam[tr]):
            a, b = tr[fam[tr] != f], tr[fam[tr] == f]
            Xa = np.concatenate([np.ones((len(idx), len(a), 1)), F[a][:, C].transpose(1, 0, 2)], 2)
            Xb = np.concatenate([np.ones((len(idx), len(b), 1)), F[b][:, C].transpose(1, 0, 2)], 2)
            coef = np.linalg.pinv(Xa) @ y[a]
            err[idx] += np.abs(np.einsum("knj,kj->kn", Xb, coef) - y[b]).sum(1)
    return err / len(tr)


def fit_apply(F, y, fam, cands, tr, te):
    """As one outer fold of ddp.lofo with the new corpora as the held-out family: choose the set by leave-one-family-
    out MAE inside the training recipes, fit it on all of them, apply to te."""
    c = cands[int(np.argmin(inner_mae(F, y, fam, cands, tr)))] if len(cands) > 1 else cands[0]
    A = np.c_[np.ones(len(tr)), F[np.ix_(tr, list(c))]]
    coef = np.linalg.lstsq(A, y[tr], rcond=None)[0]
    return np.c_[np.ones(len(te)), F[np.ix_(te, list(c))]] @ coef, c, coef


def main():
    M = json.load(open(DD / "stats" / "merged.json"))
    N = json.load(open(DC / "stats" / "merged.json"))
    S = dict(M["sources"])
    X = {k: dict(v) for k, v in M["xent"]["xent"].items()}
    new = sorted(N["sources"])
    for s in new:
        S["new_" + s] = N["sources"][s]
        X["new_" + s] = N["xent_rows"][s]
    fam_of = {r: f for f, (_, rs) in ddp.FAMILIES.items() for r in rs}
    names = sorted(fam_of)
    lv = pd.read_csv(DD / "final_levels.csv")
    T, _ = ddp.family_targets(lv, np.random.default_rng(0))
    allnames = names + ["HF:" + s for s in new]
    F, ref_cols = ddp.recipe_features(S, X, allnames)
    have = [c for c in F.columns if F[c].notna().all()]
    nonref = [c for c in have if c not in ref_cols]
    col = {c: i for i, c in enumerate(F.columns)}
    Fm = F.values
    fam = np.array([fam_of[r] for r in names] + ["NEW"] * len(new))
    tr0 = np.arange(len(names))
    te = np.arange(len(names), len(allnames))
    models = sorted({c.split("_", 2)[2] for c in ref_cols})
    pools = {"ref_lm": have, "noref": nonref}
    for m in models:
        extra = [c for c in have if c.endswith("_" + m) and c in ref_cols]
        if extra:
            pools[f"ref_{m}_only"] = nonref + extra
    out = dict(created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), corpora=LABEL,
               target_definition="bits/byte at 1B parameters (100 tokens/parameter, DataDecide ladder), equal-weight "
                                 "average over the 11 DataDecide eval sets (all11) or the 4 sets no recipe trains on (ood4)",
               training_recipes=names, features_available=dict(no_ref_lm=nonref, ref_lm=[c for c in have if c in ref_cols]),
               forecasts={}, family_heldout={})
    for tag in ("all11", "ood4"):
        y = np.r_[T[tag].loc[names, "level_1B"].values.astype(float), np.full(len(new), np.nan)]
        mm = "mismatch_ood4" if tag == "ood4" else "mismatch_all"
        P = dict(pools)
        P["mismatch"] = [mm]
        for vname, pool in P.items():
            cands = sets(pool, col) if vname != "mismatch" else [(col[mm],)]
            pred, c, coef = fit_apply(Fm, y, fam, cands, tr0, te)
            boot = []
            rng = np.random.default_rng(int(hashlib.sha256(f"{vname}|{tag}".encode()).hexdigest()[:8], 16))  # per row
            for _ in range(NBOOT):
                trb = rng.choice(tr0, len(tr0))
                if len(np.unique(fam[trb])) < 3:
                    continue
                boot.append(fit_apply(Fm, y, fam, cands, trb, te)[0])
            boot = np.array(boot)
            ranks = np.argsort(np.argsort(boot, 1), 1) + 1  # 1 = lowest loss
            order = [new[i] for i in np.argsort(pred)]
            out["forecasts"][f"{vname}|{tag}"] = dict(
                features=[F.columns[i] for i in c], coef=dict(zip(["intercept"] + [F.columns[i] for i in c], coef.tolist())),
                predicted_bpb={s: float(v) for s, v in zip(new, pred)},
                interval80={s: [float(np.quantile(boot[:, i], .1)), float(np.quantile(boot[:, i], .9))] for i, s in enumerate(new)},
                interval95={s: [float(np.quantile(boot[:, i], .025)), float(np.quantile(boot[:, i], .975))] for i, s in enumerate(new)},
                ranking=order, p_best={s: float(np.mean(ranks[:, i] == 1)) for i, s in enumerate(new)},
                median_rank={s: float(np.median(ranks[:, i])) for i, s in enumerate(new)},
                n_boot=int(len(boot)),
                # extrapolation: each chosen statistic of each corpus in units of the training recipes' SD from their
                # mean, and whether it lies outside the training recipes' range
                feature_z={s: {F.columns[i]: float((Fm[len(names) + j, i] - Fm[tr0, i].mean()) / Fm[tr0, i].std())
                               for i in c} for j, s in enumerate(new)},
                outside_training_range={s: [F.columns[i] for i in c if not Fm[tr0, i].min() <= Fm[len(names) + j, i]
                                            <= Fm[tr0, i].max()] for j, s in enumerate(new)},
                ranking_excluding_datadecide_recipes=[s for s in order if s not in OVERLAP],
                ranking_excluding_datadecide_families=[s for s in order if s not in OVERLAP and s not in RELATED])
            print(f"{vname}|{tag}: {[F.columns[i] for i in c]}  " + "  ".join(f"{s}={pred[new.index(s)]:.4f}" for s in order), flush=True)
            # each overlapping corpus forecast with its own DataDecide families removed from training
            if vname in ("ref_lm", "noref", "mismatch", "ref_gpt2_only"):
                fh = {}
                for s, d in {**OVERLAP, **RELATED}.items():
                    if s not in new:
                        continue
                    tr = tr0[~np.isin(fam[tr0], d["families"])]
                    p, cc, _ = fit_apply(Fm, y, fam, cands, tr, te)
                    fh[s] = dict(removed_families=d["families"], features=[F.columns[i] for i in cc],
                                 predicted_bpb=float(p[new.index(s)]))
                out["family_heldout"][f"{vname}|{tag}"] = fh
    # sensitivity to which runs open the sample (compression, n-gram and beta read only its start); the point models
    # are applied unchanged to statistics recomputed on reordered runs (data_choice_stats.py cpu_perm)
    pdir = DC / "stats_perm"
    ks = sorted({int(p.stem.split("_p")[-1].split("_")[0]) for p in pdir.glob("*_cpu.json")}) if pdir.exists() else []
    if ks and all((pdir / f"{s}_p{k}_cpu.json").exists() for s in new for k in ks):
        Fk = []
        for k in ks:
            Sk = dict(S)
            for s in new:
                rec = dict(json.load(open(pdir / f"{s}_p{k}_cpu.json")))
                rec.update(kind="recipe", name="HF:" + s, ref=S["new_" + s]["ref"])
                Sk["new_" + s] = rec
            Fk.append(ddp.recipe_features(Sk, X, allnames)[0])
        for key, f in out["forecasts"].items():
            coef = f["coef"]
            preds = [[f["predicted_bpb"][s] for s in new]]
            for Fp in Fk:
                preds.append([float(coef["intercept"] + sum(coef[c] * Fp.loc["HF:" + s, c] for c in f["features"])) for s in new])
            preds = np.array(preds)
            f["run_order_sensitivity"] = dict(
                orders=["sample order"] + [f"runs permuted, seed {k}" for k in ks],
                predicted_bpb={s: preds[:, i].tolist() for i, s in enumerate(new)},
                sd={s: float(preds[:, i].std(ddof=1)) for i, s in enumerate(new)},
                rankings=[[new[i] for i in np.argsort(p)] for p in preds])
    # sampling consistency: the same corpus sampled by us from HF and by DataDecide (statistics only, no outcomes)
    cons = {}
    for s, d in OVERLAP.items():
        if s in new:
            a, b = F.loc["HF:" + s], F.loc[d["recipe"]]
            cons[s] = dict(datadecide_recipe=d["recipe"], features={c: [float(a[c]), float(b[c])] for c in have})
            pr = {}
            for k, f in out["forecasts"].items():
                cols = f["features"]
                coef = f["coef"]
                pr[k] = [f["predicted_bpb"][s], float(coef["intercept"] + sum(coef[c] * b[c] for c in cols))]
            cons[s]["prediction_hf_vs_datadecide_sample"] = pr
    out["sampling_consistency"] = cons
    # rival: unigram cross-entropy of the eval text under the corpus (lower = more word overlap = better)
    U = json.load(open(DC / "stats" / "unigram_ce.json"))
    riv = {}
    for tag, evs in (("all11", None), ("ood4", ddp.OOD4)):
        ce = {s: float(np.mean([v for e, v in U["ce"][s].items() if evs is None or e in evs])) for s in new}
        riv[f"datadecide_eval|{tag}"] = dict(ce_bits_per_byte=ce, ranking=sorted(new, key=ce.get))
    for pool, ce in U.get("pools", {}).items():
        riv[pool] = dict(ce_bits_per_byte={s: ce[s] for s in new}, ranking=sorted(new, key=ce.get))
    out["rivals"] = dict(unigram_ce=dict(source="Chung & Kim 2026 (word overlap predicts performance): lower corpus->eval "
                                                "unigram cross-entropy forecasts better performance; add-0.5 smoothing",
                                         **riv))
    prov = {}
    for s in new:
        p = json.load(open(ROOT / "data" / "data_choice" / "samples" / f"{s}_provenance.json"))
        prov[s] = {k: p[k] for k in ("dataset", "revision", "note", "n_tokens", "n_docs", "sha256_npy", "created_utc")}
    out["samples"] = prov
    out["datadecide_overlap"] = {s: d for s, d in OVERLAP.items()} | {s: d for s, d in RELATED.items()}
    out["row_labels"] = ROW_LABELS
    out["primary"] = "mismatch|all11"
    json.dump(out, open(os.environ.get("DC_OUT", DC / "forecast.json"), "w"), indent=1)
    prim = DC / "forecast_primary_mismatch.json"
    if not prim.exists():  # written once, before any scoring; later runs reproduce it (per-row bootstrap seeds)
        json.dump(dict(created_utc=out["created_utc"], primary="mismatch|all11", note=ROW_LABELS["mismatch"],
                       target_definition=out["target_definition"], corpora=LABEL,
                       forecasts={k: out["forecasts"][k] for k in ("mismatch|all11", "mismatch|ood4")},
                       family_heldout={k: out["family_heldout"][k] for k in ("mismatch|all11", "mismatch|ood4")},
                       datadecide_overlap=out["datadecide_overlap"], samples=out["samples"],
                       rivals=out["rivals"]), open(prim, "w"), indent=1)


if __name__ == "__main__":
    main()
