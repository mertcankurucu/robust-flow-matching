"""Evaluate a saved core 2D model without training or rebuilding its training pool."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .models import ConditionalVectorField, VectorField
from .train import Config, generate_data, sample
from ..metrics import coverage_density, toy_metrics, wasserstein2


def evaluate(checkpoint_path, output, device="cpu", generated=None):
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    cfg = Config(**checkpoint["config"])
    cfg.device = device
    if generated is not None:
        cfg.generated = generated
    cfg.validate()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be new/empty: {output}")
    model = (
        ConditionalVectorField(dim=2, n_classes=cfg.classes, hidden_dim=cfg.hidden_dim)
        if cfg.mode == "label"
        else VectorField(dim=2, hidden_dim=cfg.hidden_dim)
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    samples, labels = sample(model, cfg)
    # Identical independent reference seeds/counts and metrics to toy.train.run.
    torch.manual_seed(cfg.seed + (10000 if cfg.mode == "target" else 20000))
    real, real_labels = generate_data(cfg, 10000)
    real, real_labels = real.cpu().numpy(), real_labels.cpu().numpy()
    if cfg.mode == "target":
        metrics = toy_metrics(real, samples, cfg.max_ot)
    else:
        per_class = []
        for c in range(cfg.classes):
            a, b = real[real_labels == c][:5000], samples[labels == c][:5000]
            per_class.append(
                {"class": c, "w2": wasserstein2(a, b), **coverage_density(a, b)}
            )
        metrics = dict(
            cw_w2=float(np.mean([r["w2"] for r in per_class])),
            cw_coverage=float(np.mean([r["coverage"] for r in per_class])),
            per_class=per_class,
        )
    from dataclasses import asdict

    result = {"config": asdict(cfg), "metrics": metrics, "evaluation_only": True}
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "samples.npz", generated=samples, generated_labels=labels
    )
    (output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf8"
    )
    print(json.dumps(metrics, indent=2), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument(
        "--generated",
        type=int,
        help="Override saved sample count; changes the evaluation protocol",
    )
    args = parser.parse_args()
    evaluate(args.checkpoint, args.output_dir, args.device, args.generated)


if __name__ == "__main__":
    main()
