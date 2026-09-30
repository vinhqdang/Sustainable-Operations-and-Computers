#!/bin/bash
# Full pipeline (resumable where noted): forecasts, tuning, tests, robustness, fairness,
# diagnostics, tables and figures.
cd "$(dirname "$0")"
set -e
step() { echo "== $1 $(date +%H:%M)"; }
# forecasts: python3 forecast.py (already run for this data snapshot)
step evalfc;     python3 evaluate_forecasts.py > ../logs/evaluate_forecasts.txt
step fcextra;    python3 forecast_extra.py > ../logs/forecast_extra.txt
step tune1;      python3 tune.py 1
python3 select_config.py 1
step tune2;      python3 tune.py 2
step tunep;      python3 tune.py p
step tunes;      python3 tune.py s
python3 select_config.py 2
for w in main trace trace_fine sens; do step $w; python3 run_test.py $w; done   # resumable
step robust;     python3 robustness.py > ../logs/robustness.txt
step fair;       python3 fairness.py && python3 fairness_analysis.py > ../logs/fairness.txt
step thm2;       python3 verify_theorem2.py > ../logs/verify_theorem2.txt
step bench;      python3 bench_solvers.py
step figs;       python3 fig_data.py && python3 analysis.py > ../logs/analysis.txt && python3 fig_theory.py
step done
