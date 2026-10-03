# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. Modifications: Robust Flow Matching authors.
# See licenses/CC-BY-NC-4.0.txt and LICENSE.
# Retained from the successful original CIFAR-10 label-noise pipeline;
# unrelated dataset/method branches were removed without changing its operations.
"""
Dataset wrapper that applies symmetric label noise to a fraction of training samples.

Symmetric label noise: with probability p, replace label y with y' drawn
uniformly from {0, ..., K-1} \\ {y}.

Usage:
    base = datasets.CIFAR10(root=..., train=True, transform=...)
    noisy = LabelNoiseDataset(base, noise_rate=0.2, num_classes=10, seed=42)
    # 20% of samples have their label replaced uniformly at random.
    # Corruption is deterministic (fixed by seed) and identical across epochs.
"""

import logging
from collections import defaultdict
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LabelNoiseDataset(Dataset):
    """Wraps a vision dataset to apply symmetric or asymmetric label noise."""

    PATRINI_PAIRS = {9: 1, 2: 0, 4: 7, 3: 5, 5: 3}

    def __init__(
        self,
        base_dataset: Dataset,
        noise_rate: float = 0.2,
        num_classes: int = 10,
        seed: int = 42,
        noise_type: str = "symmetric",
    ):
        self.base_dataset = base_dataset
        self.noise_rate = noise_rate
        self.num_classes = num_classes
        self.seed = seed
        self.noise_type = noise_type
        if hasattr(base_dataset, "targets"):
            clean_labels = list(base_dataset.targets)
        else:
            clean_labels = [base_dataset[i][1] for i in range(len(base_dataset))]
        self.clean_labels = [l if isinstance(l, int) else int(l) for l in clean_labels]
        rng = torch.Generator().manual_seed(seed)
        n = len(self.clean_labels)
        self.noisy_labels = list(self.clean_labels)
        if noise_type == "asymmetric_patrini":
            flip_mask = torch.rand(n, generator=rng) < noise_rate
            n_flipped = 0
            for idx in range(n):
                if flip_mask[idx]:
                    original = self.clean_labels[idx]
                    if original in self.PATRINI_PAIRS:
                        self.noisy_labels[idx] = self.PATRINI_PAIRS[original]
                        n_flipped += 1
        else:
            flip_mask = torch.rand(n, generator=rng) < noise_rate
            offsets = torch.randint(1, num_classes, (n,), generator=rng)
            n_flipped = 0
            for idx in range(n):
                if flip_mask[idx]:
                    original = self.clean_labels[idx]
                    self.noisy_labels[idx] = (
                        original + offsets[idx].item()
                    ) % num_classes
                    n_flipped += 1
        class_total = defaultdict(int)
        class_flipped = defaultdict(int)
        for idx in range(n):
            cl = self.clean_labels[idx]
            class_total[cl] += 1
            if self.clean_labels[idx] != self.noisy_labels[idx]:
                class_flipped[cl] += 1
        logger.info(
            f"[LabelNoiseDataset] {n_flipped}/{n} labels corrupted (rate={noise_rate}, K={num_classes}, seed={seed})"
        )
        for cls in sorted(class_total.keys()):
            logger.info(
                f"  class {cls}: {class_flipped[cls]}/{class_total[cls]} flipped"
            )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, _original_label = self.base_dataset[idx]
        return (img, self.noisy_labels[idx])
