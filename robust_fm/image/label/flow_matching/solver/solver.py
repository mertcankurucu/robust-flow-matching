# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. Modifications: Robust Flow Matching authors.
# See licenses/CC-BY-NC-4.0.txt and LICENSE.
# Retained from the successful original CIFAR-10 label-noise pipeline;
# unrelated dataset/method branches were removed without changing its operations.
from abc import ABC, abstractmethod
from torch import nn, Tensor


class Solver(ABC, nn.Module):
    """Abstract base class for solvers."""

    @abstractmethod
    def sample(self, x_0: Tensor = None) -> Tensor: ...
