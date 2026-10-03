# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Modifications: Robust Flow Matching authors.
# Core training rules adapted from the Meta example and the authors' robust extensions.
# Meta-derived code is covered by licenses/CC-BY-NC-4.0.txt.
import contextlib
import torch


def autocast(device, dtype=torch.float16):
    return (
        torch.autocast("cuda", dtype=dtype)
        if device.type == "cuda"
        else contextlib.nullcontext()
    )


def make_state(size, cfg, device):
    return {
        "scores": torch.zeros(size, device=device),
        "counts": torch.zeros(size, device=device),
        "reset": False,
    }




def train_epoch(model, loader, optimizer, scaler, state, cfg, epoch, device):
    model.train(True)
    total, seen = 0.0, 0
    for batch_index, (images, labels, idx) in enumerate(loader):
        images, labels, idx = (
            images.to(device) * 2 - 1,
            labels.to(device),
            idx.to(device),
        )
        # Preserve the CPU RNG draw from the latest class-drop implementation.
        torch.rand(1)
        conditioning = {}
        noise = torch.randn_like(images)
        t = torch.rand(len(images)).to(device)
        if cfg.method == "fm_trs":
            t = t.sqrt()
        xt = (1 - t[:, None, None, None]) * noise + t[:, None, None, None] * images
        with autocast(device):
            pred = model(xt, t, extra=conditioning)
            per_sample = ((pred - (images - noise)) ** 2).mean((1, 2, 3))
            weights = torch.ones_like(per_sample)
            if cfg.method == "fm_trs":
                with torch.no_grad():
                    state["scores"][idx] = torch.where(
                        state["counts"][idx] > 0,
                        0.999 * state["scores"][idx] + 0.001 * per_sample.detach(),
                        per_sample.detach(),
                    )
                    state["counts"][idx] += 1
                    warmup_epoch = int(0.25 * cfg.epochs)
                    # Retains the original reset AFTER the boundary's first score update.
                    if (
                        epoch == warmup_epoch
                        and batch_index == 0
                        and not state["reset"]
                    ):
                        state["scores"].zero_()
                        state["counts"].zero_()
                        state["reset"] = True
                    observed = state["counts"] > 0
                    if epoch >= warmup_epoch and observed.any():
                        threshold = torch.quantile(
                            state["scores"][observed], cfg.trs_quantile
                        )
                        weights = torch.sigmoid(
                            -(state["scores"][idx] - threshold) / cfg.tau
                        )
            loss = (weights * per_sample).mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite loss at epoch {epoch}, batch {batch_index}")
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        model.update_ema()
        total += loss.item() * len(images)
        seen += len(images)
    if not seen:
        raise ValueError("No training batches: reduce --batch-size")
    return total / seen
