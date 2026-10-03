"""CIFAR-10 training, deterministic label corruption, and clean references."""

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.datasets import CIFAR10, FakeData
from .corrupted_dataset import CorruptedDataset
from .sp_corrupted_dataset import SaltPepperDataset
from .data_transform import get_train_transform, get_score_transform


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, idx


def base_dataset(cfg, train=True, augment=False):
    transform = get_train_transform() if augment else get_score_transform()
    if cfg.smoke:
        return FakeData(
            size=40,
            image_size=(3, 32, 32),
            num_classes=10,
            transform=transform,
            random_offset=0 if train else 1000,
        )
    return CIFAR10(
        root=str(Path(cfg.data_root).expanduser()),
        train=train,
        transform=transform,
        download=False,
    )


def training_dataset(cfg, score=False):
    base = base_dataset(cfg, augment=not score)
    if cfg.corruption == "gaussian":
        base = CorruptedDataset(
            base,
            clean_fraction=1 - cfg.rate,
            noise_sigma=cfg.sigma,
            corrupt_seed=cfg.noise_seed,
        )
    else:
        base = SaltPepperDataset(
            base,
            sp_corrupt_frac=cfg.rate,
            sp_amount=0.15,
            sp_salt_vs_pepper=0.15,
            sp_seed=cfg.noise_seed,
        )
    return IndexedDataset(base)
