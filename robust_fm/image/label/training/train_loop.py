# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. Modifications: Robust Flow Matching authors.
# See licenses/CC-BY-NC-4.0.txt and LICENSE.
# Retained from the successful original CIFAR-10 label-noise pipeline;
# unrelated dataset/method branches were removed without changing its operations.
import argparse
import gc
import json
import logging
import math
from typing import Iterable
import numpy as np
import torch
from robust_fm.image.label.flow_matching.path import CondOTProbPath
from robust_fm.image.models.ema import EMA
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.aggregation import MeanMetric
from robust_fm.image.label.training.grad_scaler import NativeScalerWithGradNormCount

logger = logging.getLogger(__name__)
PRINT_FREQUENCY = 50


def _per_sample_loss_img(pred, target, loss_type="mse", huber_delta=1.0):
    return torch.pow(pred - target, 2).mean(dim=(1, 2, 3))


def compute_ccr_scores(model, score_loader, ccr_state, device, args):
    """
    CCR scoring: for each sample, evaluate the FM residual under ALL classes
    using SHARED x_t (same noise draw), so image difficulty cancels.
    Margin m_i = r_i(y_tilde) - min_{c != y_tilde} r_i(c).
    Positive margin => likely mislabeled.
    Also stores argmin class as ccr_state['best_class'].
    """
    from robust_fm.image.label.flow_matching.path import CondOTProbPath

    probe_ts = ccr_state["probe_ts"]
    K = len(probe_ts)
    M = getattr(args, "ccr_num_draws", 1)
    pool_size = ccr_state["margins"].shape[0]
    num_classes = ccr_state["num_classes"]
    decay = getattr(args, "loss_ema_decay", 0.99)
    sweep_margins = torch.zeros(pool_size, device=device)
    sweep_counts = torch.zeros(pool_size, device=device)
    sweep_residuals = torch.zeros(pool_size, num_classes, device=device)
    path = CondOTProbPath()
    was_training = model.training
    model.eval()
    inner_model = model.module if hasattr(model, "module") else model
    score_model = inner_model
    score_model.eval()
    with torch.no_grad():
        import time as _time

        n_total_batches = (
            len(score_loader) if hasattr(score_loader, "__len__") else None
        )
        score_frac = float(getattr(args, "ccr_score_fraction", 1.0))
        score_frac = max(0.0, min(1.0, score_frac))
        n_batches_to_run = n_total_batches
        if score_frac < 1.0 and n_total_batches:
            n_batches_to_run = max(1, int(score_frac * n_total_batches))
            logger.info(
                f"[CCR] Subsampling: scoring {n_batches_to_run}/{n_total_batches} batches (fraction={score_frac:.2f})"
            )
        log_every = max(1, (n_batches_to_run or 100) // 20)
        t_sweep_start = _time.time()
        batch_idx = 0
        for batch in score_loader:
            if n_batches_to_run is not None and batch_idx >= n_batches_to_run:
                break
            if len(batch) == 3:
                samples, labels, indices = batch
            else:
                continue
            samples = samples.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            indices = indices.to(device, non_blocking=True)
            samples = samples * 2.0 - 1.0
            B = samples.shape[0]
            batch_margin_sum = torch.zeros(B, device=device)
            batch_residuals_sum = torch.zeros(B, num_classes, device=device)
            for t_val in probe_ts:
                t = torch.full((B,), t_val, device=device)
                draw_margin_sum = torch.zeros(B, device=device)
                draw_residuals_sum = torch.zeros(B, num_classes, device=device)
                for _m in range(M):
                    noise = torch.randn_like(samples)
                    path_sample = path.sample(t=t, x_0=noise, x_1=samples)
                    x_t = path_sample.x_t
                    u_t = path_sample.dx_t
                    ccr_class_chunk = int(getattr(args, "ccr_class_chunk", 0))
                    if ccr_class_chunk <= 0:
                        ccr_class_chunk = max(1, min(num_classes, 2048 // max(B, 1)))
                    residuals_chunks = []
                    for c_start in range(0, num_classes, ccr_class_chunk):
                        c_end = min(c_start + ccr_class_chunk, num_classes)
                        cs = c_end - c_start
                        x_t_tile = x_t.repeat_interleave(cs, dim=0)
                        u_t_tile = u_t.repeat_interleave(cs, dim=0)
                        t_tile = t.repeat_interleave(cs, dim=0)
                        c_labels = torch.arange(c_start, c_end, device=device).repeat(B)
                        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                            pred_chunk = score_model(
                                x_t_tile, t_tile, extra={"label": c_labels}
                            )
                        r_chunk = torch.pow(pred_chunk.float() - u_t_tile, 2).mean(
                            dim=(1, 2, 3)
                        )
                        residuals_chunks.append(r_chunk.view(B, cs))
                    residuals_t = torch.cat(residuals_chunks, dim=1)
                    draw_residuals_sum += residuals_t
                    r_assigned = residuals_t[torch.arange(B, device=device), labels]
                    other_mask = torch.ones(
                        B, num_classes, dtype=torch.bool, device=device
                    )
                    other_mask[torch.arange(B, device=device), labels] = False
                    r_other_min = (
                        residuals_t.masked_fill(~other_mask, float("inf"))
                        .min(dim=1)
                        .values
                    )
                    draw_margin_sum += r_assigned - r_other_min
                batch_margin_sum += draw_margin_sum / M
                batch_residuals_sum += draw_residuals_sum / M
            sweep_margins[indices] += batch_margin_sum / K
            sweep_residuals[indices] += batch_residuals_sum / K
            sweep_counts[indices] += 1
            batch_idx += 1
            if batch_idx % log_every == 0 or batch_idx == n_batches_to_run:
                _elapsed = _time.time() - t_sweep_start
                _rate = batch_idx / max(_elapsed, 1e-06)
                if n_batches_to_run:
                    _eta = (n_batches_to_run - batch_idx) / max(_rate, 1e-06)
                    logger.info(
                        f"[CCR] batch {batch_idx}/{n_batches_to_run}  ({100.0 * batch_idx / n_batches_to_run:.1f}%)  elapsed={_elapsed:.0f}s  rate={_rate:.2f} batch/s  ETA={_eta:.0f}s"
                    )
                else:
                    logger.info(
                        f"[CCR] batch {batch_idx}  elapsed={_elapsed:.0f}s  rate={_rate:.2f} batch/s"
                    )
    if was_training:
        model.train(True)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(sweep_margins, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(sweep_counts, op=torch.distributed.ReduceOp.SUM)
        torch.distributed.all_reduce(sweep_residuals, op=torch.distributed.ReduceOp.SUM)
    observed = sweep_counts > 0
    count_col = sweep_counts.unsqueeze(1).clamp(min=1)
    sweep_avg = torch.where(
        observed,
        sweep_margins / sweep_counts.clamp(min=1),
        torch.zeros_like(sweep_margins),
    )
    avg_residuals = sweep_residuals / count_col
    best_class = avg_residuals.argmin(dim=1)
    pre_margins = ccr_state["margins"].clone()
    pre_count = ccr_state["ema_count"].clone()
    has_history = pre_count > 0
    if decay >= 1.0:
        ccr_state["margins"] = torch.where(
            has_history & observed,
            (pre_count * pre_margins + sweep_avg) / (pre_count + 1),
            torch.where(observed, sweep_avg, pre_margins),
        )
    else:
        ccr_state["margins"] = torch.where(
            has_history & observed,
            decay * pre_margins + (1 - decay) * sweep_avg,
            torch.where(observed, sweep_avg, pre_margins),
        )
    ccr_state["ema_count"] = pre_count + observed.float()
    ccr_state["best_class"] = best_class
    obs_margins = ccr_state["margins"][observed]
    ccr_state["margin_mean"] = float(obs_margins.mean().item())
    ccr_state["margin_std"] = float(obs_margins.std().item())
    logger.info(
        f"CCR scoring: K={K}, M={M}, ts={probe_ts}, observed={observed.sum().item()}/{pool_size}, margin_mean={ccr_state['margin_mean']:.5f}, margin_std={ccr_state['margin_std']:.5f}, frac_positive={(obs_margins > 0).float().mean():.3f}"
    )


def compute_ccr_diagnostics(ccr_state, epoch, output_dir=None):
    """
    Compute CCR diagnostics: AUC (margin vs is_noisy), margin separation, relabel precision.
    Requires ccr_state to contain 'is_noisy'.
    """
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None
    margins = ccr_state["margins"].cpu().numpy()
    ema_count = ccr_state["ema_count"].cpu().numpy()
    is_noisy = ccr_state["is_noisy"].cpu().numpy()
    observed = ema_count > 0
    diag = {"epoch": epoch}
    if observed.sum() < 10:
        logger.info(
            f"[CCR-diag] epoch {epoch}: too few observed ({observed.sum()}), skipping"
        )
        return diag
    m_obs = margins[observed]
    noisy_obs = is_noisy[observed]
    clean_obs = ~noisy_obs
    if noisy_obs.any() and clean_obs.any() and (roc_auc_score is not None):
        auc = roc_auc_score(noisy_obs.astype(int), m_obs)
        diag["auc"] = float(auc)
    else:
        diag["auc"] = float("nan")
    if clean_obs.any():
        diag["clean_margin_mean"] = float(m_obs[clean_obs].mean())
        diag["clean_margin_std"] = float(m_obs[clean_obs].std())
    if noisy_obs.any():
        diag["noisy_margin_mean"] = float(m_obs[noisy_obs].mean())
        diag["noisy_margin_std"] = float(m_obs[noisy_obs].std())
    if clean_obs.any() and noisy_obs.any():
        pooled_std = m_obs.std()
        diag["margin_sep"] = float(
            (diag["noisy_margin_mean"] - diag["clean_margin_mean"])
            / (pooled_std + 1e-12)
        )
    pos_mask = m_obs > 0
    diag["frac_positive"] = float(pos_mask.mean())
    if pos_mask.any():
        diag["relabel_precision"] = float(noisy_obs[pos_mask].mean())
    else:
        diag["relabel_precision"] = float("nan")
    tau = ccr_state.get("tau", 0.05)
    weights_raw = 1.0 / (1.0 + np.exp(np.clip(m_obs / tau, -500, 500)))
    diag["w_clean_mean"] = (
        float(weights_raw[clean_obs].mean()) if clean_obs.any() else float("nan")
    )
    diag["w_noisy_mean"] = (
        float(weights_raw[noisy_obs].mean()) if noisy_obs.any() else float("nan")
    )
    if output_dir is not None:
        import os

        diag_path = os.path.join(output_dir, "ccr_diagnostics.json")
        history = []
        if os.path.exists(diag_path):
            try:
                with open(diag_path, "r") as f:
                    history = json.load(f)
            except (json.JSONDecodeError, IOError):
                history = []
        history.append(diag)
        with open(diag_path, "w") as f:
            json.dump(history, f, indent=2)
        scores_dir = os.path.join(output_dir, "ccr_scores_snapshots")
        os.makedirs(scores_dir, exist_ok=True)
        npz_path = os.path.join(scores_dir, f"ccr_margins_epoch{epoch:04d}.npz")
        np.savez_compressed(
            npz_path,
            margins=margins,
            ema_count=ema_count,
            is_noisy=is_noisy,
            epoch=np.array(epoch),
        )
    return diag


def train_one_epoch(
    model: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    lr_schedule: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
    epoch: int,
    loss_scaler: NativeScalerWithGradNormCount,
    args: argparse.Namespace,
    ccr_state: dict = None,
):
    gc.collect()
    model.train(True)
    batch_loss = MeanMetric().to(device, non_blocking=True)
    epoch_loss = MeanMetric().to(device, non_blocking=True)
    accum_iter = args.accum_iter
    path = CondOTProbPath()
    getattr(args, "method", "fm") in ("fm_ccr_periodic",)
    for data_iter_step, batch in enumerate(data_loader):
        if len(batch) == 3:
            samples, labels, indices = batch
            indices = indices.to(device, non_blocking=True)
        else:
            samples, labels = batch
            indices = None
        if data_iter_step % accum_iter == 0:
            optimizer.zero_grad()
            batch_loss.reset()
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if torch.rand(1) < args.class_drop_prob:
            conditioning = {}
        else:
            conditioning = {"label": labels}
        samples = samples * 2.0 - 1.0
        noise = torch.randn_like(samples).to(device)
        t = torch.rand(samples.shape[0]).to(device)
        path_sample = path.sample(t=t, x_0=noise, x_1=samples)
        x_t = path_sample.x_t
        u_t = path_sample.dx_t
        with torch.cuda.amp.autocast():
            pred = model(x_t, t, extra=conditioning)
            if args.method in ("fm_ccr_periodic",):
                per_sample = _per_sample_loss_img(
                    pred,
                    u_t,
                    getattr(args, "loss_type", "mse"),
                    getattr(args, "huber_delta", 1.0),
                )
                if (
                    ccr_state is not None
                    and indices is not None
                    and (epoch >= ccr_state.get("start_epoch", 0))
                ):
                    observed_mask = ccr_state["ema_count"][indices] > 0
                    if observed_mask.any():
                        margins_batch = ccr_state["margins"][indices]
                        tau = ccr_state.get("tau", args.robust_tau)
                        w = torch.sigmoid(-margins_batch / tau)
                        w = torch.where(observed_mask, w, torch.ones_like(w))
                        loss = (w * per_sample).mean()
                    else:
                        loss = per_sample.mean()
                else:
                    loss = per_sample.mean()
            else:
                _lt = getattr(args, "loss_type", "mse")
                _hd = getattr(args, "huber_delta", 1.0)
                loss = _per_sample_loss_img(pred, u_t, _lt, _hd).mean()
        loss_value = loss.item()
        batch_loss.update(loss)
        epoch_loss.update(loss)
        if not math.isfinite(loss_value):
            raise ValueError(f"Loss is {loss_value}, stopping training")
        loss /= accum_iter
        apply_update = (data_iter_step + 1) % accum_iter == 0
        loss_scaler(
            loss, optimizer, parameters=model.parameters(), update_grad=apply_update
        )
        if apply_update and isinstance(model, EMA):
            model.update_ema()
        elif (
            apply_update
            and isinstance(model, DistributedDataParallel)
            and isinstance(model.module, EMA)
        ):
            model.module.update_ema()
        lr = optimizer.param_groups[0]["lr"]
        if data_iter_step % PRINT_FREQUENCY == 0:
            logger.info(
                f"Epoch {epoch} [{data_iter_step}/{len(data_loader)}]: loss = {batch_loss.compute()}, lr = {lr}"
            )
    lr_schedule.step()
    return {"loss": float(epoch_loss.compute().detach().cpu())}
