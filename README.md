# Predicting How Well and How Fast Language Models Learn from Their Data

Code for the paper "Predicting How Well and How Fast Language Models Learn from Their Data" (Computational Linguistics submission).

## Installation

Python 3.10 on CPU.

```bash
python3.10 -m venv venv && venv/bin/pip install -r requirements.txt
```

Training and the reference-model passes also need a GPU and the extra packages listed at the end of `requirements.txt`.

## Reproducing the results

```bash
PY=venv/bin/python bash reproduce.sh
```

This checks the SHA-256 of every file, runs the tests, reruns the analysis scripts from their stored inputs, redraws the paper's figures and tables and compares them with `expected/`, and checks every number in the paper against the recomputed files. It runs offline on a CPU in about 7 minutes and needs no model weights.

- `verify/paper_checks.py` checks each number in the paper against the result file and key it names.
- `scripts/figures/cl/` draws Figures 1 to 7, E.1 and F.1 and Tables F.1 and F.2; `scripts/figures/cf_numbers.py` writes Table 4 and the forecast numbers in the text.
- `scripts/verify_datadecide_headline.py` is an independent check of the DataDecide ranking that shares no code with the analysis.

## Code structure

- `exact_sources/`, `scripts/rhm/`: the exactly solvable sources and the Random Hierarchy Model.
- `text/`: tokenizers, text statistics and reference-model information curves.
- `scripts/datadecide_*.py`, `scripts/bench_mismatch_*.py`: the mismatch, recipe rankings and benchmark tests on DataDecide.
- `scripts/curve_forecast/`, `scripts/stronger_ref/`: learning-curve forecasts and larger reference models.
- `scripts/fwd_weblang/`, `scripts/codeeng/`, `scripts/extbench*`, `scripts/rate_singlepass/`: forward tests and the single-pass rate study.
- `scripts/train*.py`: the learners. Each run record in `results/` lists its arguments; `reproduce.sh` does not retrain.
- `results/`: the stored statistics, forecasts, training records and scores.

## Coverage

All 553 checked numbers and table rows agree with the recomputed files. Tables 1, 3, 5 and 6 and the appendix tables have no drawing script; their numbers are checked by `verify/paper_checks.py`. Corpora and model weights are public and not included; `data/manifest.json` lists each source.

## License

MIT. Derived data remain subject to the terms of their original releases.
