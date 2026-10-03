# Robust Flow Matching under Target Corruption and Label Noise

Official code for our **NeurIPS 2026** paper, with OT-FM, FM-TRS, and FM-CCS
experiments on 2D distributions and CIFAR-10. Temporal Residual Score (TRS)
downweights corrupted targets using residuals across sampled times.
Comparative Conditional Score (CCS) downweights unreliable labels by comparing
residuals across candidate classes.

## Setup

Activate a Python 3.10+ environment and run from the cloned repository's root.
Use the same interpreter for installation and experiments; skip installation
if your environment already has the dependencies. Commands below use Bash
(Linux/macOS, or WSL on Windows).

```bash
export PYTHON_BIN="$(python -c 'import sys; print(sys.executable)')"
"$PYTHON_BIN" -m pip install -e ".[cifar]"
export DATA_ROOT=/path/to/CIFAR10  # contains cifar-10-batches-py/
export OUTPUT="$PWD/results/full"
```

2D data are generated locally; `DATA_ROOT` and metric weights are only needed for CIFAR-10. CIFAR-10 data must already exist (e.g. via torchvision's `CIFAR10(root, download=True)`); the scripts do not download it.

```bash
"$PYTHON_BIN" scripts/prepare_cifar10.py --data-root "$DATA_ROOT" --cache-metrics
```

## Run

Run these commands directly on your machine. 2D supports CPU or CUDA
(evaluation needs about 6 GB RAM).
CIFAR-10 runs require an NVIDIA GPU with CUDA-enabled PyTorch.

```bash
# 2D: target outliers and label noise, four models.
DEVICE=cpu bash scripts/run_2d.sh  # Use DEVICE=cuda for GPU training.

# CIFAR-10: 20% Gaussian target corruption, OT-FM and TRS, 1000 epochs.
bash scripts/run_cifar10.sh target

# CIFAR-10: 40% symmetric label noise, OT-FM and CCS, 1000 epochs.
# CCS runs at temperatures 0.001 and 0.01.
bash scripts/run_cifar10.sh label

# Collect results.
"$PYTHON_BIN" scripts/summarize_results.py --root "$OUTPUT"
```

Use a fresh output directory for new experiments.

For Slurm clusters, [slurm/](slurm/) contains the batch templates used on our
cluster. Adapt the account/partition, resource requests, time limit, and module
setup to your system before submitting with `sbatch` from the repository root.
Set `LOAD_MODULES=0` to use an already configured Python/CUDA environment.
Each array runs all six tasks (2D plus five CIFAR-10 models).

## Outputs

- Numbers: `$OUTPUT/summary.md` (also CSV and JSON).
- 2D figure: `$OUTPUT/2d/Figures/spirals_figure_y_v2.{pdf,png}`.
- CIFAR-10 metrics and samples: `$OUTPUT/cifar_target_*/evaluation/` and
  `$OUTPUT/cifar_label_*/evaluate/`.

## Key files

- [robust_fm/toy/train.py](robust_fm/toy/train.py): 2D training, TRS scores/weights, and CCS margins/weights.
- [robust_fm/image/train.py](robust_fm/image/train.py): CIFAR-10 target training and TRS scores/weights.
- [robust_fm/image/label/training/train_loop.py](robust_fm/image/label/training/train_loop.py): CIFAR-10 CCS scoring and weighted training.
- [robust_fm/image/label/run.py](robust_fm/image/label/run.py): CIFAR-10 label-noise training and evaluation entry point.

## Citation

```bibtex
@inproceedings{kurucu2026robust,
  title     = {Robust Flow Matching under Target Corruption and Label Noise},
  author    = {Kurucu, Mert Can and Englesson, Erik and Azizpour, Hossein},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026}
}
```

## License

Original contributions: MIT. Meta-derived portions: CC BY-NC 4.0.
See [LICENSE](LICENSE) and [upstream licenses](licenses/).
