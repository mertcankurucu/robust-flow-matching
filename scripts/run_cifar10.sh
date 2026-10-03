#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OUTPUT=${OUTPUT:-$PWD/results/full}
case "${1:-label}" in
  target) tasks=(1 2) ;;
  label) tasks=(3 4 5) ;;
  *) echo 'Usage: bash scripts/run_cifar10.sh target|label' >&2; exit 2 ;;
esac
for task in "${tasks[@]}"; do bash scripts/run_case.sh "$task"; done
"${PYTHON_BIN:-python}" scripts/summarize_results.py --root "$OUTPUT" --allow-incomplete
