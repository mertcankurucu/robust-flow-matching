"""CIFAR-10 architecture shared with the target-corruption experiments."""

from ..model import create_model


def instantiate_model(architechture, is_discrete, use_ema, num_classes):
    if (architechture, is_discrete, use_ema, num_classes) != (
        "cifar10",
        False,
        True,
        10,
    ):
        raise ValueError("This release supports the continuous CIFAR-10 EMA model")
    return create_model()
