# Carbon-aware anticipatory scheduling of deferrable computing workloads

Code, data and manuscript for the study *"Anticipating demand, not only carbon: receding-horizon
scheduling of deferrable computing workloads across data centres with calibrated carbon-intensity
forecasts"*, prepared for **Sustainable Operations and Computers** (KeAi).

## Contents

| Path | Description |
|---|---|
| `data/raw/gb_regional_ci_2022_2024.csv.gz` | Half-hourly regional carbon intensity and generation mix for Great Britain, 2022-2024, from the NESO Carbon Intensity API |
| `src/fetch_data.py` | Downloads the raw data from the API |
| `src/common.py` | Sites, facility parameters, data splits, hourly loader |
| `src/forecast.py` | Direct per-horizon gradient-boosted forecaster (1-24 h) and benchmarks |
| `src/conformal.py` | Normalised split-conformal upper quantiles with adaptive conformal inference |
| `src/sim.py` | Workload generator, min-cost-flow planning core, baseline policies, CARMA, oracle, simulator |
| `src/experiment.py` | Experiment harness (forecast cubes, weeks, policy factory) |
| `src/tune.py`, `src/select_config.py` | Hyperparameter tuning on 2023-Q4 validation weeks |
| `src/run_test.py` | Out-of-sample evaluation on the 52 weeks of 2024 and sensitivity analyses |
| `src/evaluate_forecasts.py`, `src/fig_data.py`, `src/analysis.py` | Tables, statistics and figures |
| `results/` | All raw experiment outputs (CSV/JSON) and summary tables |
| `figures/` | Figures used in the manuscript (PDF) |
| `manuscript/` | Elsevier `elsarticle` LaTeX source, bibliography, highlights and compiled PDF |

## Reproducing

```bash
pip install -r requirements.txt
cd src
python fetch_data.py            # optional: the raw file is already included
python forecast.py              # trains the forecaster, writes data/forecasts.pkl
python evaluate_forecasts.py    # forecast accuracy and calibration tables
python tune.py 1                # tuning stage 1 (anticipation weight, look-ahead)
python tune.py 2 1.0 12         # tuning stage 2 (carbon cost view) for the stage-1 winner
python select_config.py         # writes results/carma_config.json
python run_test.py main         # 2024 test: 6 scenarios x 52 weeks x 10 policies
python run_test.py sens         # sensitivity analyses
python fig_data.py && python analysis.py
cd ../manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

The full pipeline runs in about two hours on four CPU cores. All random seeds are fixed.

## Data licence

Carbon-intensity data are provided by the National Energy System Operator (NESO) Carbon Intensity
API (https://carbonintensity.org.uk/) under CC BY 4.0.
