#!/usr/bin/env bash
#SBATCH --job-name=core_full
#SBATCH --output=core_full_%A_%a.out
#SBATCH --error=core_full_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --array=0-5%6

# Full runs. Account/partition come from sbatch; no fixed node request.
# full_ampere.sh shares this workload with different module defaults.
set -euo pipefail
cd "${REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}"
export PYTHON_BIN=${PYTHON_BIN:?Set the absolute Python path of your existing environment}
export OUTPUT=${OUTPUT:-$PWD/results/full_${SLURM_ARRAY_JOB_ID}}
export TORCH_HOME=${TORCH_HOME:-$HOME/.cache/torch}
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-${SLURM_CPUS_ON_NODE:-16}}
export MKL_NUM_THREADS=$OMP_NUM_THREADS
mkdir -p "$OUTPUT"
finish() {
  local run_status=$?
  trap - EXIT
  local report_status=0
  flock "$OUTPUT/.summary.lock" "$PYTHON_BIN" scripts/summarize_results.py \
    --root "$OUTPUT" --allow-incomplete || report_status=$?
  if [[ "$run_status" -ne 0 ]]; then exit "$run_status"; fi
  exit "$report_status"
}
trap finish EXIT
if [[ "${LOAD_MODULES:-1}" == 1 ]]; then
  module purge
  read -ra modules <<< "${BERZELIUS_MODULES:-buildtool-easybuild/5.1.2-hpc8c13cca44 buildenv-gcccuda/12.9.1-gcc11 Miniforge3/25.3.1-0}"
  module load "${modules[@]}"
fi
"$PYTHON_BIN" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; print("PyTorch",torch.__version__,"CUDA",torch.version.cuda,"GPU",torch.cuda.get_device_name(0))'
bash scripts/run_case.sh "${SLURM_ARRAY_TASK_ID:?Expected array task 0..5}"
