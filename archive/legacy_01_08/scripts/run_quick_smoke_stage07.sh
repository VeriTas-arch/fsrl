#!/usr/bin/env bash
set -euo pipefail

# Example smoke tests. Run from repository root.
python sources/07_modular_ablation_active_rank/ablation_suite.py --smoke --timeout 800 --big-timeout 1800 --output-dir outputs_smoke_ablation
