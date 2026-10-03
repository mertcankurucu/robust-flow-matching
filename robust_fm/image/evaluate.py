# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modifications: Robust Flow Matching authors.
# Evaluation adapted from Meta's image example and the authors' metric helpers.
# Meta-derived code is covered by licenses/CC-BY-NC-4.0.txt.
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from .data import base_dataset
from .edm_time_discretization import get_time_discretization
from .train import autocast
from ..metrics import coverage_density


@torch.no_grad()
def generate(model, cfg, count, device, class_id=None, ode_steps=50, labels=None):
    from torchdiffeq import odeint

    x0 = torch.randn(count, 3, 32, 32, device=device)
    if labels is None:
        labels = (
            torch.arange(count, device=device) % 10
            if class_id is None
            else torch.full((count,), class_id, dtype=torch.long, device=device)
        )
    extra = {}

    def velocity(t, x):
        with autocast(device):
            return model(x, torch.zeros(len(x), device=device) + t, extra=extra).float()

    # Retains the source's EDM-inspired grid, including its nonzero start time.
    grid = get_time_discretization(ode_steps).to(device)
    generated = odeint(velocity, x0, grid, method="heun2")[-1]
    return torch.floor((generated * 0.5 + 0.5).clamp(0, 1) * 255).float() / 255


def evaluate(
    model,
    cfg,
    output,
    device,
    *,
    generated=50000,
    per_class=0,
    reference_split="train",
    batch_size=64,
    ode_steps=50,
):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.manual_seed(cfg.seed + 10000)
    np.random.seed(cfg.seed + 10000)
    if cfg.smoke:
        images = generate(model, cfg, 10, device, ode_steps=2)
        save_image(images, output / "SMOKE_TEST_samples.png", nrow=5)
        result = {
            "smoke_test": True,
            "note": "Synthetic data/tiny model; no scientific metrics computed.",
        }
        (output / "metrics.json").write_text(
            json.dumps(result, indent=2), encoding="utf8"
        )
        return result

    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.inception import InceptionScore
    from .features import InceptionFeatureExtractor

    real_data = base_dataset(cfg, train=reference_split == "train")
    real_loader = DataLoader(
        real_data, batch_size=batch_size, shuffle=False, num_workers=cfg.workers
    )
    fid = FrechetInceptionDistance(normalize=True).to(device)
    inception_score = InceptionScore(normalize=True).to(device)
    features = InceptionFeatureExtractor(device)
    real_features, real_labels, fake_features = [], [], []
    for images, labels in real_loader:
        images = images.to(device)
        fid.update(images, real=True)
        real_features.append(features.extract(images))
        real_labels.append(labels.numpy())
    real_features, real_labels = (
        np.concatenate(real_features),
        np.concatenate(real_labels),
    )
    for start in range(0, generated, batch_size):
        count = min(batch_size, generated - start)
        # Follow the clean reference's labels, as the original evaluation loop did.
        labels = torch.as_tensor(
            real_labels[np.arange(start, start + count) % len(real_labels)],
            device=device,
        )
        images = generate(model, cfg, count, device, ode_steps=ode_steps, labels=labels)
        fid.update(images, real=False)
        inception_score.update(images)
        fake_features.append(features.extract(images))
        if start == 0:
            save_image(images[:64], output / "samples.png", nrow=8)
        if start % (batch_size * 50) == 0:
            print(
                f"Evaluation: {start + len(images)}/{generated} generated", flush=True
            )
    fake_features = np.concatenate(fake_features)
    is_mean, is_std = inception_score.compute()
    result = {
        "fid": float(fid.compute()),
        "is_mean": float(is_mean),
        "is_std": float(is_std),
        "reference_split": reference_split,
        "real_count": len(real_data),
        "generated_count": generated,
        "ode_intervals": ode_steps,
        "ode_method": "heun2",
        "seed": cfg.seed + 10000,
    }
    rng = np.random.RandomState(42)

    def subset(x):
        return x if len(x) <= 10000 else x[rng.choice(len(x), 10000, replace=False)]

    result.update(coverage_density(subset(real_features), subset(fake_features)))
    del inception_score, fake_features
    (output / "metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf8"
    )
    return result
