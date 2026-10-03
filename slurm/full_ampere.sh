#!/usr/bin/env bash
#SBATCH --job-name=core_ampere
#SBATCH --output=core_ampere_%A_%a.out
#SBATCH --error=core_ampere_%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --array=0-5%6

# Submit from a Berzelius Ampere login node, with your account passed to sbatch.
# One full A100 per task; no node list, GPU memory constraint, or reservation.
# Training, evaluation and summary generation are shared with full_hopper.sh.
set -euo pipefail
export REPO=${REPO:-${SLURM_SUBMIT_DIR:?Submit from the repository root}}
export BERZELIUS_MODULES=${BERZELIUS_MODULES:-"buildtool-easybuild/4.3.3-nscf4a947 buildenv-gcccuda/12.1.1-gcc12.3.0 Miniforge3/24.7.1-2-hpc1-bdist"}
export LOAD_MODULES=${LOAD_MODULES:-1}
export OUTPUT=${OUTPUT:-$REPO/results/full_ampere_${SLURM_ARRAY_JOB_ID}}
source "$REPO/slurm/full_hopper.sh"
