"""Recompute the analysis result files (forward tests, larger references, reanalyses, single-pass rate study, Random
Hierarchy Model, compute frontier and the later analyses) from the shipped inputs, in a scratch copy of the repository, and
compare each output with the shipped file. Numbers must agree to 1e-9 (relative); everything else must be identical.

    python verify/regenerate.py [--work DIR]   # DIR (kept) holds the regenerated tree; paper_checks.py --root DIR reads it

CPU only, no network, a few minutes. Exit status 1 on any difference or failed script.
"""
import filecmp
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable
REF = "results/reanalysis/"
CF = "results/curve_forecast/"

# (command, outputs compared with the shipped copies, environment); in dependency order
STEPS = [
    ("scripts/reanalysis/nl_table.py", [REF + "nl_table.json"], {}),
    ("scripts/reanalysis/nl_dev_ngram_level.py", [REF + "nl_dev_ngram_level.json"], {}),
    ("scripts/reanalysis/nl_level_maps.py", [REF + "nl_level_maps.json"], {}),
    ("scripts/reanalysis/nl_ngram_route.py", [REF + "nl_ngram_route.json"], {}),
    ("scripts/reanalysis/level_drop_split.py", [REF + "level_drop_split.json"], {}),
    ("scripts/reanalysis/rate_law_fixed_size.py", [REF + "rate_law_fixed_size.json"], {}),
    ("scripts/reanalysis/rate_law_fixed_size_b.py", [REF + "rate_law_fixed_size_b.json"], {}),
    ("scripts/reanalysis/revised_forecaster.py", [REF + "revised_forecaster.json"], {}),
    ("scripts/reanalysis/revised_boot.py", [REF + "revised_boot.json"], {}),
    ("scripts/reanalysis/shrink_fit.py", [REF + "shrink_fit.json"], {}),
    ("scripts/stronger_ref/score.py", [CF + "stronger_ref/score.json"], {}),
    ("scripts/stronger_ref/shape_test.py", [CF + "stronger_ref/shape_test.json"], {}),
    ("scripts/stronger_ref/score.py --sub7", [CF + "stronger_ref7b/score.json"], {}),
    ("scripts/stronger_ref/shape_test.py --sub7", [CF + "stronger_ref7b/shape_test.json"], {}),
    ("scripts/stronger_ref/score_half.py", [CF + "stronger_ref14b/score_half500.json"], {"TAG14": "q3_14b"}),
    ("scripts/reanalysis/stronger_baselines.py", [REF + "stronger_baselines.json"], {}),
    ("scripts/reanalysis/ref_subset_cost.py", [REF + "ref_subset_cost.json"], {}),
    ("scripts/reanalysis/matched_compute_ratios.py", [REF + "matched_compute_ratios.json"], {}),
    ("scripts/new_corpora_analysis.py", [CF + "new_corpora/analysis.json"], {}),
    ("scripts/reanalysis/compute_frontier.py", [REF + "compute_frontier.json"], {}),
    ("scripts/figures/fig_ed_compute.py", ["figures/fig_ed_compute.pdf", "figures/fig_ed_compute.png"], {}),
    ("scripts/reanalysis/mismatch_vs_word.py", [REF + "mismatch_vs_word.json"], {}),
    ("scripts/reanalysis/law_applicability_check.py", [REF + "law_applicability_check.json"], {}),
    ("scripts/extbench/score.py", ["results/extbench/score.json"], {}),
    ("scripts/reanalysis/shape_vs_pooled.py", [REF + "shape_vs_pooled.json"], {}),
    ("scripts/reanalysis/shape_within_kind.py", [REF + "shape_within_kind.json"], {}),
    ("scripts/fwd_weblang/score.py", ["results/fwd_weblang/score.json"], {}),
    ("scripts/rate_singlepass/analyze.py", ["results/rate_singlepass/analysis.json"], {}),
    ("scripts/rate_singlepass/analyze.py --dir results/rate_singlepass25m",
     ["results/rate_singlepass25m/analysis.json"], {}),
    ("scripts/rate_singlepass/window_slopes.py results/rate_singlepass25m --family language",
     ["results/rate_singlepass25m/window_slopes_language.json"], {}),
    ("scripts/reanalysis/fall_diagnosis.py", [REF + "fall_diagnosis.json"], {}),
    ("scripts/codeeng/score_t1.py", ["results/code25m/score.json"], {}),
    ("scripts/codeeng/score_t2.py", ["results/fwd_english/score.json"], {}),
    ("scripts/reanalysis/compression_fall.py", [REF + "compression_fall.json"], {}),
    ("scripts/rhm/analyze.py", ["results/rhm/analysis.json"], {}),
    # benchmark swaps, curve increment, held-out code, open-sci-ref forecast, compute-matched comparison, applicability
    # rule, sensitivity to protocol choices, floor conventions of the increment, code level and fall, source-disjoint test
    ("scripts/task_reversals.py", ["results/task_reversals/reversals_k2.json",
                                   "results/task_reversals/reversals_k3.json"], {}),
    ("scripts/reanalysis/infocurve_increment.py", ["results/infocurve_increment/increment.json"], {}),
    ("scripts/reanalysis/code_heldout.py", ["results/code_heldout/score.json"], {}),
    ("scripts/opensciref_forecast/score.py", ["results/opensciref_forecast/scores.json"], {}),
    ("scripts/costmatch/consolidate.py", ["results/costmatch/summary.json", "results/costmatch/rows.json"], {}),
    ("scripts/applicability/calib.py", ["results/applicability/calibration.json"], {}),
    ("scripts/applicability/score.py", ["results/applicability/score.json"], {}),
    ("scripts/sensitivity/datadecide_floors.py", ["results/sensitivity/datadecide_floors.json"], {}),
    ("scripts/sensitivity/singlepass_floors.py", ["results/sensitivity/singlepass_floors.json"], {}),
    ("scripts/sensitivity/sens.py", ["results/sensitivity/sens.json"], {}),
    ("scripts/sensitivity/key_table.py", ["results/sensitivity/key_table.json"], {}),
    ("scripts/increment_floor/increment_floor.py", ["results/increment_floor/score.json"], {}),
    ("scripts/code_decomp/decomp.py", ["results/code_decomp/decomp.json"], {}),
    ("scripts/source_disjoint/score.py", ["results/source_disjoint/score.json"], {}),
]
# Parts of fall_diagnosis.json that need the 20,000-target reference loss files (about 1 GB, not shipped): the document
# bootstrap of each corpus's own fall and the document count of each corpus (Q2 per corpus and reliability), and the
# block built on them. These blocks are carried over from the shipped file into the regenerated tree, so the later steps
# (applicability, code level and fall) and the four document counts printed in the article read shipped values.
SKIP = {REF + "fall_diagnosis.json": (("Q2", "per_corpus"), ("Q2", "reliability"), ("exploratory_reliable_x",), ("*", "boot_sd_x")),
        # check that the full 20,000-target curve rebuilt from those files equals the stored curve (needs the same files)
        "results/sensitivity/sens.json": (("ref1000", "full_reproduces_stored_max_abs"),)}


def diff(a, b, path, out, skip):
    if path in skip or (path and ("*", path[-1]) in skip):
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            if k not in a or k not in b:
                if path + (k,) not in skip and ("*", k) not in skip:
                    out.append("/".join(map(str, path + (k,))) + " missing")
            else:
                diff(a[k], b[k], path + (k,), out, skip)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            diff(x, y, path + (i,), out, skip)
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
        if not (a == b or (math.isnan(a) and math.isnan(b)) or abs(a - b) <= 1e-9 * max(1.0, abs(a), abs(b))):
            out.append(f"{'/'.join(map(str, path))}: {a!r} != {b!r}")
    elif a != b:
        out.append(f"{'/'.join(map(str, path))}: {str(a)[:60]!r} != {str(b)[:60]!r}")


def carry(shipped, new, path, skip):
    """Copy the skipped blocks (not recomputable from the shipped files) from the shipped file into the regenerated one, so
    that later steps and the manuscript checks read the shipped values of those blocks."""
    if path in skip or (path and ("*", path[-1]) in skip):
        return shipped
    if isinstance(shipped, dict) and isinstance(new, dict):
        out = dict(new)
        for k, v in shipped.items():
            if k in new:
                out[k] = carry(v, new[k], path + (k,), skip)
            elif path + (k,) in skip or ("*", k) in skip:
                out[k] = v
        return out
    if isinstance(shipped, list) and isinstance(new, list) and len(shipped) == len(new):
        return [carry(x, y, path + (i,), skip) for i, (x, y) in enumerate(zip(shipped, new))]
    return new


def main():
    keep = "--work" in sys.argv
    work = sys.argv[sys.argv.index("--work") + 1] if keep else tempfile.mkdtemp(prefix="regen_")
    os.makedirs(work, exist_ok=True)
    for d in ("scripts", "results", "text", "exact_sources", "data"):
        if os.path.exists(os.path.join(ROOT, d)):
            shutil.copytree(os.path.join(ROOT, d), os.path.join(work, d))
    for d in ("figures", "paper"):
        os.makedirs(os.path.join(work, d), exist_ok=True)
    # shape_vs_pooled.py and shape_within_kind.py are shipped byte for byte (their SHA-256 is recorded in
    # results/fwd_weblang/fall_line.json) and use the earlier names of two result directories; alias them in the scratch copy
    for old, new in (("referee_reanalysis", "reanalysis"), ("extbench_2026-10-28", "extbench")):
        os.symlink(new, os.path.join(work, "results", old))
    env = dict(os.environ, PYTHONHASHSEED="0", MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    bad = 0
    for cmd, outs, extra in STEPS:
        t = time.time()
        r = subprocess.run([PY] + cmd.split(), cwd=work, env=dict(env, **extra), capture_output=True, text=True)
        msg = []
        if r.returncode != 0:
            msg.append("script failed: " + (r.stderr.strip().splitlines() or ["?"])[-1])
        for o in outs:
            shipped = os.path.join(ROOT, "expected", o) if o.startswith("figures/") else os.path.join(ROOT, o)
            new = os.path.join(work, o)
            if not os.path.exists(new):
                msg.append(f"{o} not written"); continue
            if o.endswith(".json"):
                out = []
                diff(json.load(open(shipped)), json.load(open(new)), (), out, set(SKIP.get(o, ())))
                if o in SKIP:
                    json.dump(carry(json.load(open(shipped)), json.load(open(new)), (), set(SKIP[o])), open(new, "w"), indent=1)
                if out:
                    msg.append(f"{o}: {len(out)} differences, e.g. {out[:3]}")
            elif not filecmp.cmp(shipped, new, shallow=False):
                msg.append(f"{o} differs from the shipped copy")
        bad += bool(msg)
        print(f"{'FAIL' if msg else 'same'}  {time.time() - t:5.1f}s  {cmd}" + "".join(f"\n      {m}" for m in msg), flush=True)
    print(f"regenerate: {len(STEPS)} scripts rerun, {len(STEPS) - bad} reproduce their shipped outputs, {bad} do not")
    if not keep:
        shutil.rmtree(work)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
