"""
Dataset wrapper that deterministically corrupts a subset of training images.

Usage:
    base = datasets.CIFAR10(root=..., train=True, transform=...)
    corrupted = CorruptedDataset(base, clean_fraction=0.1, noise_sigma=0.5, corrupt_seed=42)
    # 10% of images returned clean, 90% with Gaussian noise (sigma=0.5 on [0,1] scale)
    # The clean/noisy split is fixed by corrupt_seed — identical across method runs.
"""

import logging

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class CorruptedDataset(Dataset):
    """Wraps a vision dataset to add Gaussian noise to a deterministic subset."""

    def __init__(
        self,
        base_dataset: Dataset,
        clean_fraction: float = 0.0,
        noise_sigma: float = 0.5,
        corrupt_seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.noise_sigma = noise_sigma

        n = len(base_dataset)
        n_clean = round(clean_fraction * n)

        # Deterministic split — same seed always gives the same clean indices
        rng = torch.Generator().manual_seed(corrupt_seed)
        perm = torch.randperm(n, generator=rng)
        self.clean_indices = set(perm[:n_clean].tolist())

        n_noisy = n - n_clean
        logger.info(
            f"[CorruptedDataset] {n_clean} clean + {n_noisy} noisy "
            f"(sigma={noise_sigma}, seed={corrupt_seed})"
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]
        if idx not in self.clean_indices:
            # Add Gaussian noise on [0,1] scale (before train_loop's [-1,1] scaling)
            img = img + self.noise_sigma * torch.randn_like(img)
            img = img.clamp(0.0, 1.0)
        return img, label
