#!/usr/bin/env bash
set -euo pipefail
python ablation_suite.py --smoke --timeout 600 --big-timeout 1200 --output-dir ablation_runs_smoke_new
