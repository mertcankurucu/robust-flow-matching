"""
Dataset wrapper that applies salt-and-pepper noise to a per-class subset of training images.

Usage:
    base = datasets.CIFAR10(root=..., train=True, transform=...)
    corrupted = SaltPepperDataset(base, sp_corrupt_frac=0.1, sp_amount=0.1,
                                   sp_salt_vs_pepper=0.5, sp_seed=42)
    # 10% of images PER CLASS get salt-and-pepper noise (10% of pixels flipped).
    # The corrupted index set is fixed by sp_seed — identical across method runs.
    # The noise pattern per image is also deterministic (sp_seed + idx).
"""

import logging
from collections import defaultdict

import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SaltPepperDataset(Dataset):
    """Wraps a vision dataset to apply salt-and-pepper noise to a per-class subset."""

    def __init__(
        self,
        base_dataset: Dataset,
        sp_corrupt_frac: float = 0.1,
        sp_amount: float = 0.1,
        sp_salt_vs_pepper: float = 0.5,
        sp_seed: int = 42,
    ):
        self.base_dataset = base_dataset
        self.sp_amount = sp_amount
        self.sp_salt_vs_pepper = sp_salt_vs_pepper
        self.sp_seed = sp_seed

        # Build per-class index lists
        if hasattr(base_dataset, "targets"):
            targets = base_dataset.targets
        else:
            # Fallback: iterate dataset (slow but universal)
            targets = [base_dataset[i][1] for i in range(len(base_dataset))]

        class_indices = defaultdict(list)
        for idx, label in enumerate(targets):
            label_val = label if isinstance(label, int) else int(label)
            class_indices[label_val].append(idx)

        # Deterministic per-class corruption selection
        rng = torch.Generator().manual_seed(sp_seed)
        self.corrupted_indices = set()

        total_clean = 0
        total_corrupt = 0
        for cls in sorted(class_indices.keys()):
            indices = class_indices[cls]
            n_corrupt = int(sp_corrupt_frac * len(indices))
            perm = torch.randperm(len(indices), generator=rng)
            for p in perm[:n_corrupt].tolist():
                self.corrupted_indices.add(indices[p])
            total_corrupt += n_corrupt
            total_clean += len(indices) - n_corrupt

        logger.info(
            f"[SaltPepperDataset] {total_clean} clean + {total_corrupt} corrupted "
            f"(amount={sp_amount}, salt_vs_pepper={sp_salt_vs_pepper}, seed={sp_seed})"
        )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img, label = self.base_dataset[idx]

        if idx in self.corrupted_indices:
            # Deterministic per-image RNG so noise pattern is fixed across epochs
            rng = torch.Generator().manual_seed(self.sp_seed + idx)
            # Which pixels to corrupt
            mask = torch.rand(img.shape, generator=rng) < self.sp_amount
            # Which of those are salt (1) vs pepper (0)
            salt = torch.rand(img.shape, generator=rng) < self.sp_salt_vs_pepper
            img = img.clone()
            img[mask & salt] = 1.0
            img[mask & ~salt] = 0.0

        return img, label
