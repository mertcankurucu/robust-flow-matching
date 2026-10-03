# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. Modifications: Robust Flow Matching authors.
# See licenses/CC-BY-NC-4.0.txt and LICENSE.
# Retained from the successful original CIFAR-10 label-noise pipeline;
# unrelated dataset/method branches were removed without changing its operations.
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as datasets
from robust_fm.image.label.model import instantiate_model
from robust_fm.image.label.training import distributed_mode
from robust_fm.image.label.training.data_transform import get_train_transform
from robust_fm.image.label.training.eval_loop import eval_model, eval_per_class_metrics
from robust_fm.image.label.training.grad_scaler import (
    NativeScalerWithGradNormCount as NativeScaler,
)
from robust_fm.image.label.training.load_and_save import load_model, save_model
from robust_fm.image.label.training.train_loop import (
    train_one_epoch,
    compute_ccr_scores,
    compute_ccr_diagnostics,
)

logger = logging.getLogger(__name__)


class IndexedDataset(torch.utils.data.Dataset):
    """Wraps a dataset to also return the sample index."""

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        return (*data, idx)


def main(args):
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(levelname)-8s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    distributed_mode.init_distributed_mode(args)
    logger.info("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
    logger.info("{}".format(args).replace(", ", ",\n"))
    if distributed_mode.is_main_process():
        args_filepath = Path(args.output_dir) / "args.json"
        logger.info(f"Saving args to {args_filepath}")
        with open(args_filepath, "w") as f:
            json.dump(vars(args), f)
    device = torch.device(args.device)
    seed = args.seed + distributed_mode.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True
    if getattr(args, "data_root", "") and args.data_root:
        args.data_path = os.path.join(args.data_root, args.dataset)
        logger.info(f"Using data_root: {args.data_root} -> data_path: {args.data_path}")
    _image_size = getattr(args, "image_size", 0) or 0
    logger.info(
        f"Initializing Dataset: {args.dataset} (image_size={_image_size or 'native'})"
    )
    transform_train = get_train_transform(image_size=_image_size)
    dataset_train = datasets.CIFAR10(
        root=args.data_path, train=True, download=False, transform=transform_train
    )
    logger.info(dataset_train)
    dataset_eval = None
    if getattr(args, "label_noise_rate", 0.0) > 0:
        from robust_fm.image.label.training.label_noise_dataset import LabelNoiseDataset

        if dataset_eval is None:
            dataset_eval = dataset_train
        dataset_train = LabelNoiseDataset(
            dataset_train,
            noise_rate=args.label_noise_rate,
            num_classes=args.num_classes or 10,
            seed=args.label_noise_seed,
            noise_type=getattr(args, "label_noise_type", "symmetric"),
        )
    pool_size = None
    needs_indices = args.method == "fm_ccr_periodic"
    if needs_indices:
        pool_size = len(dataset_train)
        dataset_train = IndexedDataset(dataset_train)
        logger.info(
            f"CCS mode: wrapped dataset with IndexedDataset (pool_size={pool_size})"
        )
    logger.info("Intializing DataLoader")
    num_tasks = distributed_mode.get_world_size()
    global_rank = distributed_mode.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    logger.info(str(sampler_train))
    data_loader_eval = None
    if dataset_eval is not None:
        sampler_eval = torch.utils.data.DistributedSampler(
            dataset_eval, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        data_loader_eval = torch.utils.data.DataLoader(
            dataset_eval,
            sampler=sampler_eval,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
    logger.info("Initializing Model")
    _arch = getattr(args, "model_arch", "") or args.dataset
    model = instantiate_model(
        architechture=_arch,
        is_discrete=args.discrete_flow_matching,
        use_ema=args.use_ema,
        num_classes=getattr(args, "num_classes", None),
    )
    model.to(device)
    model_without_ddp = model
    logger.info(str(model_without_ddp))
    eff_batch_size = (
        args.batch_size * args.accum_iter * distributed_mode.get_world_size()
    )
    logger.info(f"Learning rate: {args.lr:.2e}")
    logger.info(f"Accumulate grad iterations: {args.accum_iter}")
    logger.info(f"Effective batch size: {eff_batch_size}")
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True
        )
        model_without_ddp = model.module
    optimizer = torch.optim.AdamW(
        model_without_ddp.parameters(), lr=args.lr, betas=args.optimizer_betas
    )
    if args.decay_lr:
        lr_schedule = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            total_iters=args.epochs,
            start_factor=1.0,
            end_factor=1e-08 / args.lr,
        )
    else:
        lr_schedule = torch.optim.lr_scheduler.ConstantLR(
            optimizer, total_iters=args.epochs, factor=1.0
        )
    logger.info(f"Optimizer: {optimizer}")
    logger.info(f"Learning-Rate Schedule: {lr_schedule}")
    loss_scaler = NativeScaler()
    load_model(
        args=args,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer,
        loss_scaler=loss_scaler,
        lr_schedule=lr_schedule,
    )
    ccr_state = None
    ccr_score_loader = None
    if getattr(args, "method", "fm") in ("fm_ccr_periodic",) and pool_size is not None:
        num_classes = args.num_classes or 10
        ccr_probe_ts_str = getattr(args, "ccr_probe_ts", "0.5")
        ccr_probe_ts = [float(x.strip()) for x in ccr_probe_ts_str.split(",")]
        inner_ds = (
            dataset_train.dataset
            if hasattr(dataset_train, "dataset")
            else dataset_train
        )
        if hasattr(inner_ds, "clean_labels") and hasattr(inner_ds, "noisy_labels"):
            ccr_is_noisy = torch.tensor(
                [c != n for c, n in zip(inner_ds.clean_labels, inner_ds.noisy_labels)],
                dtype=torch.bool,
                device=device,
            )
        else:
            ccr_is_noisy = torch.zeros(pool_size, dtype=torch.bool, device=device)
        ccr_state = {
            "margins": torch.zeros(pool_size, device=device),
            "best_class": torch.zeros(pool_size, dtype=torch.long, device=device),
            "ema_count": torch.zeros(pool_size, device=device),
            "probe_ts": ccr_probe_ts,
            "num_classes": num_classes,
            "is_noisy": ccr_is_noisy,
            "tau": args.robust_tau,
            "start_epoch": getattr(args, "ccr_start_epoch", 0),
            "margin_mean": 0.0,
            "margin_std": 1.0,
            "tau_norm": getattr(args, "tau_norm", 1.0),
            "q_norm": getattr(args, "q_norm", 0.0),
            "candidate_classifier": None,
            "candidate_normalize": None,
        }
        from robust_fm.image.label.training.data_transform import (
            get_score_transform as _get_score_tf,
        )

        ccr_score_transform = _get_score_tf(image_size=_image_size)
        ccr_score_base = datasets.CIFAR10(
            root=args.data_path,
            train=True,
            download=False,
            transform=ccr_score_transform,
        )
        if getattr(args, "label_noise_rate", 0.0) > 0:
            from robust_fm.image.label.training.label_noise_dataset import (
                LabelNoiseDataset as _LND,
            )

            ccr_score_base = _LND(
                ccr_score_base,
                noise_rate=args.label_noise_rate,
                num_classes=num_classes,
                seed=args.label_noise_seed,
            )
        ccr_score_ds = IndexedDataset(ccr_score_base)
        ccr_score_sampler = torch.utils.data.DistributedSampler(
            ccr_score_ds, num_replicas=num_tasks, rank=global_rank, shuffle=False
        )
        _ccr_loader_bs = 2048
        ccr_score_loader = torch.utils.data.DataLoader(
            ccr_score_ds,
            sampler=ccr_score_sampler,
            batch_size=_ccr_loader_bs,
            num_workers=0,
            pin_memory=args.pin_mem,
            drop_last=False,
        )
        logger.info(
            f"CCR state: probe_ts={ccr_probe_ts}, num_classes={num_classes}, start_epoch={ccr_state['start_epoch']}, tau={args.robust_tau}, normalize_margin={getattr(args, 'ccr_normalize_margin', False)}, tau_norm={getattr(args, 'tau_norm', 1.0)}, q_norm={getattr(args, 'q_norm', 0.0)}, relabel={getattr(args, 'ccr_relabel', False)}, is_noisy={ccr_is_noisy.sum().item()}/{pool_size}"
        )
        if getattr(args, "method", "fm") == "fm_ccr_periodic":
            _fracs_str = getattr(args, "ccr_refresh_fracs", "0.50,0.75")
            _fracs = [float(x.strip()) for x in _fracs_str.split(",")]
            _refresh = sorted({int(f * args.epochs) for f in _fracs})
            ccr_state["refresh_epochs"] = _refresh
            ccr_state["start_epoch"] = _refresh[0]
            logger.info(
                f"fm_ccr_periodic: score refresh at epochs {_refresh} (fracs={_fracs})"
            )
    logger.info(f"Start from {args.start_epoch} to {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        ccr_start = ccr_state["start_epoch"] if ccr_state is not None else 0
        ccr_score_every = getattr(args, "ccr_score_every", 5)
        _do_ccr_score = False
        if (
            ccr_state is not None
            and ccr_score_loader is not None
            and (not args.eval_only)
        ):
            if getattr(args, "method", "fm") == "fm_ccr_periodic":
                _do_ccr_score = epoch in ccr_state.get("refresh_epochs", [])
            else:
                _do_ccr_score = epoch >= ccr_start and epoch % ccr_score_every == 0
        if _do_ccr_score:
            logger.info(f"[CCR] Running scoring sweep at epoch {epoch}")
            if args.distributed:
                ccr_score_loader.sampler.set_epoch(epoch)
            compute_ccr_scores(model, ccr_score_loader, ccr_state, device, args)
            if (
                getattr(args, "method", "fm") == "fm_ccr_periodic"
                and distributed_mode.is_main_process()
            ):
                _diag = compute_ccr_diagnostics(
                    ccr_state,
                    epoch,
                    output_dir=args.output_dir if args.output_dir else None,
                )
                logger.info(f"[CCR-periodic] Refresh at epoch {epoch}: {_diag}")
        if not args.eval_only:
            train_stats = train_one_epoch(
                model=model,
                data_loader=data_loader_train,
                optimizer=optimizer,
                lr_schedule=lr_schedule,
                device=device,
                epoch=epoch,
                loss_scaler=loss_scaler,
                args=args,
                ccr_state=ccr_state,
            )
            log_stats = {
                **{f"train_{k}": v for k, v in train_stats.items()},
                "epoch": epoch,
            }
            ccr_diag_freq = getattr(args, "ccr_inline_score_freq", 5)
            if (
                ccr_state is not None
                and getattr(args, "method", "fm") != "fm_ccr_periodic"
                and distributed_mode.is_main_process()
                and (epoch >= ccr_start)
                and (epoch % ccr_diag_freq == 0)
            ):
                ccr_diag = compute_ccr_diagnostics(
                    ccr_state,
                    epoch,
                    output_dir=args.output_dir if args.output_dir else None,
                )
                log_stats.update(
                    {f"ccr_{k}": v for k, v in ccr_diag.items() if k != "epoch"}
                )
        else:
            log_stats = {"epoch": epoch}
        if args.output_dir and (
            args.eval_frequency > 0
            and (epoch + 1) % args.eval_frequency == 0
            or args.eval_only
        ):
            if not args.eval_only:
                save_model(
                    args=args,
                    model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer,
                    lr_schedule=lr_schedule,
                    loss_scaler=loss_scaler,
                    epoch=epoch,
                )
            eval_loader = (
                data_loader_eval if data_loader_eval is not None else data_loader_train
            )
            if args.distributed:
                eval_loader.sampler.set_epoch(0)
            if distributed_mode.is_main_process():
                fid_samples = args.fid_samples - (num_tasks - 1) * (
                    args.fid_samples // num_tasks
                )
            else:
                fid_samples = args.fid_samples // num_tasks
            eval_stats = eval_model(
                model,
                eval_loader,
                device,
                epoch=epoch,
                fid_samples=fid_samples,
                args=args,
            )
            log_stats.update({f"eval_{k}": v for k, v in eval_stats.items()})
            if (
                getattr(args, "eval_per_class", 0) > 0
                and distributed_mode.is_main_process()
            ):
                pc_stats = eval_per_class_metrics(
                    model, eval_loader, device, epoch=epoch, args=args
                )
                if "cas" in pc_stats.get("overall", {}):
                    log_stats["eval_cas"] = pc_stats["overall"]["cas"]
                if "cwfid_mean" in pc_stats.get("overall", {}):
                    log_stats["eval_cwfid_mean"] = pc_stats["overall"]["cwfid_mean"]
                if "cw_coverage" in pc_stats.get("overall", {}):
                    log_stats["eval_cw_coverage"] = pc_stats["overall"]["cw_coverage"]
        if args.output_dir and distributed_mode.is_main_process():
            with open(
                os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8"
            ) as f:
                f.write(json.dumps(log_stats) + "\n")
        if args.eval_only:
            break
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info(f"Training time {total_time_str}")
