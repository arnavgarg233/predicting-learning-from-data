#!/usr/bin/env bash
# Reproduce the paper's figures, tables and numbers from the stored results, offline on CPU (about 7 minutes).
#   bash reproduce.sh                        # PY=python3 by default
#   PY=/path/to/python bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python3}
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT
mkdir -p figures paper regenerated

echo "== 1/7 checksums of the shipped files (MANIFEST.sha256)"
"$PY" verify/check_manifest.py

echo "== 2/7 exactness tests of the solvable sources"
for t in tests/test_*.py; do "$PY" "$t" > /dev/null; echo "passed $t"; done

echo "== 3/7 independent check of the DataDecide headline (shares no code with the analysis)"
"$PY" scripts/verify_datadecide_headline.py

echo "== 4/7 figures and tables"
"$PY" scripts/figures/cf_numbers.py
"$PY" scripts/figures/cl/table_corpora.py
for s in fig1 fig2 fig3 fig4 fig5; do echo "-- $s"; "$PY" "scripts/figures/cl/$s.py"; done

echo "== 5/7 derived result files, recomputed into regenerated/"
"$PY" scripts/curve_forecast/score_merged.py 80 --out regenerated/score80.json
"$PY" scripts/curve_forecast/score_merged.py stage3 --out regenerated/score_stage3.json
"$PY" scripts/evalset_rate_bootstrap.py --out regenerated/evalset_rate_test.json
"$PY" scripts/datadecide_rate_variance_checks.py --out regenerated/rate_variance_checks.json
"$PY" scripts/midscale_rate_verify.py --out regenerated/midscale_rate_verify.json
"$PY" scripts/midscale_best_cells.py --out regenerated/best_cells.txt
"$PY" scripts/limisiewicz_refit.py --out regenerated/limisiewicz2026_refit.json
# benchmark-text test, scored in a scratch copy from the stored extract of DataDecide's downstream results
BM="$WORK/bench"
mkdir -p "$BM/results"
cp -R scripts "$BM/"
cp -R results/bench_mismatch results/bench_mismatch_score results/datadecide "$BM/results/"
rm -f "$BM/results/bench_mismatch_score/score.json" "$BM/results/bench_mismatch_score/outcomes_1B_final_by_seed.csv"
(cd "$BM" && "$PY" scripts/bench_mismatch_score.py "$BM/no_cache" > /dev/null \
  && "$PY" scripts/bench_mismatch_proxy.py "$BM/no_cache" primary > /dev/null \
  && "$PY" scripts/bench_mismatch_proxy.py "$BM/no_cache" continuous > /dev/null)
cp "$BM/results/bench_mismatch_score/score.json" regenerated/bench_mismatch_score.json
cp "$BM/results/bench_mismatch_score/outcomes_1B_final_by_seed.csv" regenerated/outcomes_1B_final_by_seed.csv
"$PY" verify/check_replay.py

echo "== 6/7 every analysis script rerun from its stored inputs and compared with its stored output"
"$PY" verify/regenerate.py --work "$WORK/regen"

echo "== 7/7 numbers printed in the manuscript, read from the recomputed files"
"$PY" verify/paper_checks.py --root "$WORK/regen"

echo "PASS: checksums, tests, headline check, figures and tables, regenerated results and manuscript numbers all agree"
