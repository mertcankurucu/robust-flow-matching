# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. Modifications: Robust Flow Matching authors.
# See licenses/CC-BY-NC-4.0.txt and LICENSE.
# Retained from the successful original CIFAR-10 label-noise pipeline;
# unrelated dataset/method branches were removed without changing its operations.
"""CAS using an existing CIFAR-10 ResNet-18 classifier state dictionary."""

import logging
import torch
import torch.nn as nn
from torchvision import models, transforms

logger = logging.getLogger(__name__)
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.201)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
CIFAR10_NORMALIZE = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)


def _make_resnet18(num_classes: int):
    """ResNet-18 adapted for small images (32-64 px) and arbitrary class count."""
    model = models.resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def make_cifar10_resnet18():
    """Backward-compat alias."""
    return _make_resnet18(10)


def load_classifier(
    checkpoint_path: str,
    device: torch.device,
    num_classes: int,
    normalize_mean=CIFAR10_MEAN,
    normalize_std=CIFAR10_STD,
):
    """Generic CAS classifier loader.

    Returns (model, normalize_transform). Inputs to the model should be in [0,1].
    """
    model = _make_resnet18(num_classes)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    normalize = transforms.Normalize(normalize_mean, normalize_std)
    logger.info(
        f"Loaded CAS classifier from {checkpoint_path} (num_classes={num_classes}, normalize_mean={normalize_mean})"
    )
    return (model, normalize)


def load_cifar10_classifier(checkpoint_path: str, device: torch.device):
    """Backward-compat: returns just the model (uses CIFAR-10 normalize at call site)."""
    model, _ = load_classifier(
        checkpoint_path,
        device,
        num_classes=10,
        normalize_mean=CIFAR10_MEAN,
        normalize_std=CIFAR10_STD,
    )
    return model


@torch.no_grad()
def compute_cas_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    classifier: nn.Module,
    normalize=CIFAR10_NORMALIZE,
) -> tuple:
    """
    Compute CAS on a batch.

    Args:
        images: [B, 3, H, W] in [0, 1]
        labels: [B] int class indices
        classifier: pretrained ResNet-18
        normalize: transforms.Normalize matching the classifier's training stats

    Returns:
        (num_correct, num_total) for this batch
    """
    normalized = normalize(images)
    preds = classifier(normalized).argmax(dim=1)
    correct = (preds == labels).sum().item()
    return (correct, labels.shape[0])
