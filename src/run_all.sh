#!/bin/bash
# Full revision pipeline (resumable): main, trace and sensitivity tests, robustness, fairness.
cd "$(dirname "$0")"
for w in main trace sens; do echo "== $w $(date +%H:%M)"; python3 run_test.py $w || exit 1; done
echo "== robust $(date +%H:%M)"; python3 robustness.py > /dev/null || exit 1
echo "== fair $(date +%H:%M)"; python3 fairness.py || exit 1
echo "== done $(date +%H:%M)"
