#!/usr/bin/env bash
set -euo pipefail
python ablation_suite.py --timeout 600 --big-timeout 1200 --output-dir ablation_runs_full
