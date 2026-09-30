# Carbon-aware anticipatory scheduling of deferrable computing workloads

Code, data and manuscript for the study *"Anticipating demand in carbon-aware scheduling across data
centres: guarantees and fair carbon attribution"*, prepared for **Sustainable Operations and Computers** (KeAi).

## Contents

| Path | Description |
|---|---|
| `data/raw/gb_regional_ci_2022_2024.csv.gz` | Half-hourly regional carbon intensity and generation mix for Great Britain, 2022-2024, from the NESO Carbon Intensity API |
| `src/fetch_data.py` | Downloads the raw data from the API (days with missing half-hours are re-requested in one-day chunks) |
| `src/prepare_trace.py` | Builds the job table of the Alibaba 2018 trace from the raw batch-task table |
| `src/common.py` | Sites, facility parameters, data splits, hourly loader |
| `src/forecast.py` | Direct per-horizon gradient-boosted forecaster (1-24 h) and benchmarks |
| `src/conformal.py` | Normalised split-conformal upper quantiles with adaptive conformal inference |
| `src/sim.py` | Workload generator, min-cost-flow planning core, baseline policies (incl. Scenario-MPC), CARMA, oracle, simulator |
| `src/trace.py` | Trace-derived workload (batching, deadlines, origins) |
| `src/experiment.py` | Experiment harness (forecast cubes, weeks, policy factory) |
| `src/tune.py`, `src/select_config.py` | Hyperparameter tuning on 2023-Q4 validation weeks |
| `src/run_test.py` | Out-of-sample evaluation on the 52 weeks of 2024 and sensitivity analyses |
| `src/robustness.py` | Controlled prediction-error experiments |
| `src/verify_theorem2.py` | Numerical check of Theorem 2 in one-day episodes with a backstop |
| `src/fairness.py`, `src/fairness_analysis.py` | Pooling game among sites: coalition values, dual (core) carbon attribution, stability checks (Theorem 3) |
| `src/evaluate_forecasts.py`, `src/fig_data.py`, `src/analysis.py`, `src/fig_theory.py` | Tables, statistics and figures |
| `results/` | All raw experiment outputs (CSV/JSON) and summary tables |
| `figures/` | Figures used in the manuscript (PDF) |
| `manuscript/` | Elsevier `elsarticle` LaTeX source (theory in `sec_theory.tex`, proofs in `appendix_proofs.tex`), bibliography, highlights and compiled PDF |

## Reproducing

```bash
pip install -r requirements.txt      # pinned versions (Python 3.11)
cd src
python fetch_data.py                 # optional: the raw file is already included
python prepare_trace.py              # optional: rebuilds data/traces/alibaba2018_jobs.csv.gz
                                     # from batch_task.csv of cluster-trace-v2018 (not included)
./run_all.sh                         # complete pipeline, see below
cd ../manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

`run_all.sh` runs, in order:

| Step | Command | Output |
|---|---|---|
| Forecaster | `python forecast.py` | `data/forecasts.pkl` |
| Forecast evaluation | `python evaluate_forecasts.py`, `python forecast_extra.py` | accuracy, calibration, pinball, Diebold-Mariano, cost-view offsets |
| Tuning, stage 1 | `python tune.py 1`, then `python select_config.py 1` | anticipation weight and look-ahead per regime |
| Tuning, stages 2, p, s | `python tune.py 2`, `python tune.py p`, `python tune.py s` | cost view, MPC-Protect, Scenario-MPC |
| Configuration | `python select_config.py 2` | `results/carma_config.json` |
| Tests | `python run_test.py main`, `trace`, `trace_fine`, `sens` | `results/test_*.csv` |
| Robustness | `python robustness.py` | `results/robustness*.{csv,json}` |
| Fairness | `python fairness.py`, `python fairness_analysis.py` | `results/fairness_*` |
| Theorem 2 check | `python verify_theorem2.py` | `results/verify_theorem2*.csv` |
| Solver benchmark | `python bench_solvers.py` | `results/bench_solvers.json` |
| Tables and figures | `python fig_data.py`, `python analysis.py`, `python fig_theory.py` | `results/*.tex`, `figures/*.pdf` |

The main experiment runs 12 policies (including the oracle and ablations) on 52 weeks x 3 loads x
2 regimes x 3 workload traces; Scenario-MPC is run in the spatio-temporal regime only.
`run_test.py` caches finished runs in `results/partial_<experiment>_<hash>.jsonl`, where the hash
covers all source files, the configuration and the input data, so that a change to any of them
starts a fresh run. Delete these files for a clean rerun.

The results in the paper were produced on 4 cores of an Intel Xeon processor at 2.1 GHz (Linux,
Python 3.11, single-threaded solvers). The full pipeline takes several hours; decision times
depend on the hardware. All random seeds are fixed.

## Data licence

Carbon-intensity data are provided by the National Energy System Operator (NESO) Carbon Intensity
API (https://carbonintensity.org.uk/) under CC BY 4.0.
