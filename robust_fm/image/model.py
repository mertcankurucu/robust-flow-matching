# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modifications: Robust Flow Matching authors.
# Derived from Meta's CIFAR-10 example; see licenses/CC-BY-NC-4.0.txt.
from .models.ema import EMA
from .models.unet import UNetModel


def create_model(smoke=False):
    config = dict(
        in_channels=3,
        model_channels=128,
        out_channels=3,
        num_res_blocks=4,
        attention_resolutions=[2],
        dropout=0.3,
        channel_mult=[2, 2, 2],
        conv_resample=False,
        dims=2,
        num_classes=10,
        use_checkpoint=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_new_attention_order=True,
        with_fourier_features=False,
    )
    if smoke:
        config.update(model_channels=32, num_res_blocks=1, channel_mult=[1, 2])
    return EMA(UNetModel(**config))
