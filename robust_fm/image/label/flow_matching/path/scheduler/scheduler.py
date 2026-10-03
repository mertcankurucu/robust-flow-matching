# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. Modifications: Robust Flow Matching authors.
# See licenses/CC-BY-NC-4.0.txt and LICENSE.
# Retained from the successful original CIFAR-10 label-noise pipeline;
# unrelated dataset/method branches were removed without changing its operations.
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import torch
from torch import Tensor


@dataclass
class SchedulerOutput:
    """Represents a sample of a conditional-flow generated probability path.

    Attributes:
        alpha_t (Tensor): :math:`\\alpha_t`, shape (...).
        sigma_t (Tensor): :math:`\\sigma_t`, shape (...).
        d_alpha_t (Tensor): :math:`\\frac{\\partial}{\\partial t}\\alpha_t`, shape (...).
        d_sigma_t (Tensor): :math:`\\frac{\\partial}{\\partial t}\\sigma_t`, shape (...).

    """

    alpha_t: Tensor = field(metadata={"help": "alpha_t"})
    sigma_t: Tensor = field(metadata={"help": "sigma_t"})
    d_alpha_t: Tensor = field(metadata={"help": "Derivative of alpha_t."})
    d_sigma_t: Tensor = field(metadata={"help": "Derivative of sigma_t."})


class Scheduler(ABC):
    """Base Scheduler class."""

    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        """
        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\\alpha_t,\\sigma_t,\\frac{\\partial}{\\partial t}\\alpha_t,\\frac{\\partial}{\\partial t}\\sigma_t`
        """
        ...

    @abstractmethod
    def snr_inverse(self, snr: Tensor) -> Tensor:
        """
        Computes :math:`t` from the signal-to-noise ratio :math:`\\frac{\\alpha_t}{\\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...


class ConvexScheduler(Scheduler):
    @abstractmethod
    def __call__(self, t: Tensor) -> SchedulerOutput:
        """Scheduler for convex paths.

        Args:
            t (Tensor): times in [0,1], shape (...).

        Returns:
            SchedulerOutput: :math:`\\alpha_t,\\sigma_t,\\frac{\\partial}{\\partial t}\\alpha_t,\\frac{\\partial}{\\partial t}\\sigma_t`
        """
        ...

    @abstractmethod
    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        """
        Computes :math:`t` from :math:`\\kappa_t`.

        Args:
            kappa (Tensor): :math:`\\kappa`, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        ...

    def snr_inverse(self, snr: Tensor) -> Tensor:
        """
        Computes :math:`t` from the signal-to-noise ratio :math:`\\frac{\\alpha_t}{\\sigma_t}`.

        Args:
            snr (Tensor): The signal-to-noise, shape (...)

        Returns:
            Tensor: t, shape (...)
        """
        kappa_t = snr / (1.0 + snr)
        return self.kappa_inverse(kappa=kappa_t)


class CondOTScheduler(ConvexScheduler):
    """CondOT Scheduler."""

    def __call__(self, t: Tensor) -> SchedulerOutput:
        return SchedulerOutput(
            alpha_t=t,
            sigma_t=1 - t,
            d_alpha_t=torch.ones_like(t),
            d_sigma_t=-torch.ones_like(t),
        )

    def kappa_inverse(self, kappa: Tensor) -> Tensor:
        return kappa
