#!/usr/bin/env bash
# One full core condition. No installation or downloads.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN=${PYTHON_BIN:-python}
OUTPUT=${OUTPUT:-$PWD/results/full}
SEED=${SEED:-0}
case "${1:?Expected task index 0..5}" in
  0)
    "$PYTHON_BIN" -m robust_fm.toy.figure --device "${DEVICE:-cuda}" \
      --seed "$SEED" --steps 10001 --distance 4.0 --output-dir "$OUTPUT/2d"
    exit 0 ;;
  1) mode=target; method=ot_fm ;;
  2) mode=target; method=fm_trs ;;
  3) mode=label; method=ot_fm; tau=0.001; name=ot_fm ;;
  4) mode=label; method=fm_ccs; tau=0.001; name=fm_ccs_tau0.001 ;;
  5) mode=label; method=fm_ccs; tau=0.01; name=fm_ccs_tau0.01 ;;
  *) echo 'Expected task index 0..5' >&2; exit 2 ;;
esac
DATA_ROOT=${DATA_ROOT:?Set the existing parent of cifar-10-batches-py}
"$PYTHON_BIN" -c 'import sys; from pathlib import Path; from robust_fm.image.label.run import preflight; preflight(Path(sys.argv[1]), sys.argv[2])' "$DATA_ROOT" "${CAS_CKPT:-}"
if [[ "$mode" == target ]]; then
  out="$OUTPUT/cifar_target_$method"
  resume=()
  if [[ -f "$out/checkpoint.pt" ]]; then resume=(--resume "$out/checkpoint.pt"); fi
  "$PYTHON_BIN" -m robust_fm.image.run train --method "$method" \
    --rate 0.2 --corruption gaussian --sigma 0.3 --tau 0.01 --seed "$SEED" \
    --epochs 1000 --batch-size 64 --workers 8 --save-every 1 --skip-eval \
    --data-root "$DATA_ROOT" --output-dir "$out" "${resume[@]}"
  "$PYTHON_BIN" -m robust_fm.image.run evaluate --checkpoint "$out/checkpoint.pt" \
    --data-root "$DATA_ROOT" --output-dir "$out/evaluation" \
    --reference-split train --generated 50000 --per-class 0 --ode-steps 50
else
  out="$OUTPUT/cifar_label_$name"
  launcher=("$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node=1 -m robust_fm.image.label.run)
  # Label training cannot resume, so a rerun skips finished phases.
  if [[ -f "$out/train/completed.json" ]]; then
    echo "Training already complete: $out/train"
  else
    probes=()
    if [[ -n "${CCS_PROBE_TIMES:-}" ]]; then probes=(--probe-times "$CCS_PROBE_TIMES"); fi
    "${launcher[@]}" train --method "$method" --tau "$tau" --seed "$SEED" \
      --data-root "$DATA_ROOT" --output-dir "$out/train" "${probes[@]}"
  fi
  if [[ -f "$out/evaluate/completed.json" ]]; then
    echo "Evaluation already complete: $out/evaluate"
  else
    # Keep an interrupted evaluation for inspection; evaluate into a clean directory.
    if [[ -d "$out/evaluate" ]]; then
      mv "$out/evaluate" "$out/evaluate_incomplete_$(date +%Y%m%d_%H%M%S)"
    fi
    classifier=()
    if [[ -n "${CAS_CKPT:-}" ]]; then classifier=(--cas-classifier "$CAS_CKPT"); fi
    "${launcher[@]}" evaluate --checkpoint "$out/train/checkpoint.pth" \
      --data-root "$DATA_ROOT" --output-dir "$out/evaluate" "${classifier[@]}"
  fi
fi
