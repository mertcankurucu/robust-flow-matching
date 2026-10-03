# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# licenses/CC-BY-NC-4.0.txt file in the repository root.
import torch
from torchvision.transforms.v2 import (
    Compose,
    RandomHorizontalFlip,
    Resize,
    ToDtype,
    ToImage,
)


def get_train_transform(image_size: int = 0):
    """Training transform. If image_size > 0, prepend a Resize."""
    transform_list = [ToImage()]
    if image_size > 0:
        transform_list.append(Resize((image_size, image_size), antialias=True))
    transform_list += [
        RandomHorizontalFlip(),
        ToDtype(torch.float32, scale=True),
    ]
    return Compose(transform_list)


def get_score_transform(image_size: int = 0):
    """Score loader transform (no augmentation). If image_size > 0, prepend Resize."""
    transform_list = [ToImage()]
    if image_size > 0:
        transform_list.append(Resize((image_size, image_size), antialias=True))
    transform_list.append(ToDtype(torch.float32, scale=True))
    return Compose(transform_list)
