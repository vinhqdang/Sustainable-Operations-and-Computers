#!/bin/bash
# Multi-grid evaluation: forecaster, forecast evaluation, tuning and test for every fleet.
# Usage: ./run_fleets.sh [fleet ...]   (default: EU US BR GLOBAL); prerequisites: grids_fetch.py eu us br nz
cd "$(dirname "$0")"
set -e
FLEETS=${@:-EU US BR GLOBAL}
for f in $FLEETS; do
  export FLEET=$f
  echo "== $f forecast $(date +%H:%M)"
  [ -f ../data/forecasts_$(echo $f | tr A-Z a-z).pkl ] || python3 forecast.py > ../logs/forecast_$f.log
  echo "== $f evalfc $(date +%H:%M)"; python3 evaluate_forecasts.py > ../logs/evalfc_$f.txt
  echo "== $f tune $(date +%H:%M)";   python3 fleet_run.py tune > ../logs/tune_$f.log
  echo "== $f test $(date +%H:%M)";   python3 fleet_run.py test > ../logs/test_$f.log
done
echo "== done $(date +%H:%M)"
