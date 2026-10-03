#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OUTPUT=${OUTPUT:-$PWD/results/full}
bash scripts/run_case.sh 0
"${PYTHON_BIN:-python}" scripts/summarize_results.py --root "$OUTPUT" --allow-incomplete
