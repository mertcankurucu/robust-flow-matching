# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modifications: Robust Flow Matching authors.
# Adapted from the image evaluation helpers; see licenses/CC-BY-NC-4.0.txt.
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import inception_v3, Inception_V3_Weights


class InceptionFeatureExtractor:
    """Extract pool3 features (2048-d) from InceptionV3 for kNN metrics."""

    def __init__(self, device: torch.device):
        self.device = device
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights)
        # Remove the final FC layer — we want pool features
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(device)
        # Preprocessing: resize to 299×299, torchvision/ImageNet preprocessing (distinct from the FID backbone)
        self.transform = weights.transforms()

    @torch.no_grad()
    def extract(self, images: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        """images: (N, 3, H, W) in [0, 1].  Returns (N, 2048) numpy.

        Processes images in chunks of ``batch_size`` to avoid OOM on large N
        (e.g. per-class CW-coverage with 5000 images at 299x299).
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        feats_list = []
        for i in range(0, images.shape[0], batch_size):
            chunk = images[i : i + batch_size].to(self.device, non_blocking=True)
            chunk = F.interpolate(
                chunk, size=(299, 299), mode="bilinear", align_corners=False
            )
            chunk = (chunk - mean) / std
            f = self.model(chunk)
            if isinstance(f, tuple):
                f = f[0]  # InceptionOutputs namedtuple
            feats_list.append(f.cpu().numpy())
            del chunk, f
        return np.concatenate(feats_list, axis=0)
