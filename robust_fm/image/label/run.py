"""CIFAR-10 OT-FM and periodic CCS, using the original label-noise engine.

Launch with torch.distributed.run --standalone --nproc_per_node=1. Training
starts from scratch; evaluation accepts the original checkpoint.pth format.
"""

import argparse
import importlib.metadata
import json
import math
import os
from pathlib import Path


def configuration(method, tau=0.001):
    return dict(
        dataset="cifar10",
        method="fm" if method == "ot_fm" else "fm_ccr_periodic",
        num_classes=10,
        batch_size=64,
        epochs=1000,
        accum_iter=1,
        lr=1e-4,
        optimizer_betas=[0.9, 0.95],
        decay_lr=False,
        use_ema=True,
        pure_conditional=True,
        class_drop_prob=0.0,
        cfg_scale=0.0,
        label_noise_rate=0.4,
        label_noise_seed=42,
        label_noise_type="symmetric",
        seed=0,
        # OT-FM does not use robust scores; retain its recorded metadata too.
        robust_tau=0.0 if method == "ot_fm" else tau,
        loss_ema_decay=0.99 if method == "ot_fm" else 0.999,
        ccr_probe_ts="0.5"
        if method == "ot_fm"
        else "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        ccr_refresh_fracs="0.50,0.75" if method == "ot_fm" else "0.25,0.50",
        ccr_num_draws=1,
        ccr_class_chunk=0,
        loss_type="mse",
        image_size=0,
        discrete_flow_matching=False,
        eval_frequency=100,
        compute_fid=False,
        eval_only=False,
        eval_per_class=0,
        fid_samples=50000,
        save_fid_samples=False,
        edm_schedule=True,
        ode_method="euler",
        ode_options={"nfe": 100},
        device="cuda",
        num_workers=10,
        pin_mem=True,
        start_epoch=0,
        resume="",
        dist_on_itp=False,
        dist_url="env://",
        world_size=1,
        data_root="",
        cas_classifier="",
    )


def preflight(data_root, cas_classifier=""):
    """Use existing data and cached metric weights; never download from a job."""
    import torch
    from torchvision.datasets import CIFAR10

    dataset = CIFAR10(root=str(data_root), train=True, download=False)
    if len(dataset) != 50000:
        raise ValueError("Expected the complete 50,000-image CIFAR-10 training set")
    cache = Path(torch.hub.get_dir()) / "checkpoints"
    for name in (
        "weights-inception-2015-12-05-6726825d.pth",
        "inception_v3_google-0cc3c7bd.pth",
    ):
        if not (cache / name).is_file():
            raise FileNotFoundError(
                f"Metric weights missing: {cache / name}. Set TORCH_HOME to your existing cache or run scripts/prepare_cifar10.py explicitly."
            )
    if cas_classifier:
        from .training.cas_utils import load_cifar10_classifier

        load_cifar10_classifier(cas_classifier, torch.device("cpu"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["train", "evaluate"])
    parser.add_argument("--method", choices=["ot_fm", "fm_ccs"], default="fm_ccs")
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Parent of existing cifar-10-batches-py/",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="For evaluation only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument(
        "--cas-classifier",
        type=Path,
        help="Existing CIFAR-10 ResNet-18 state dictionary",
    )
    parser.add_argument(
        "--probe-times",
        default=None,
        help="Comma-separated probe times; default: 0.0,0.1,...,1.0",
    )
    args = parser.parse_args()
    if args.tau <= 0 or args.workers < 0:
        parser.error("Temperature must be positive and workers nonnegative")
    if os.environ.get("RANK") != "0" or os.environ.get("WORLD_SIZE") != "1":
        parser.error(
            "Use: python -m torch.distributed.run --standalone --nproc_per_node=1 -m robust_fm.image.label.run ..."
        )
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error(f"Output is not empty: {args.output_dir}. Choose a new directory.")
    import torch

    if not torch.cuda.is_available():
        parser.error("The original CIFAR-10 label pipeline requires a CUDA GPU")
    cas = str(args.cas_classifier.resolve()) if args.cas_classifier else ""
    preflight(args.data_root, cas)
    cfg = configuration(args.method, args.tau)
    cfg.update(seed=args.seed, num_workers=args.workers)
    if args.probe_times is not None:
        probes = [float(t) for t in args.probe_times.split(",")]
        if not probes or any(not 0 <= t <= 1 for t in probes):
            parser.error("Probe times must be in [0, 1]")
        cfg["ccr_probe_ts"] = args.probe_times
    if args.command == "evaluate":
        if args.checkpoint is None or not args.checkpoint.is_file():
            parser.error("Evaluation requires --checkpoint")
        checkpoint = torch.load(
            str(args.checkpoint), map_location="cpu", weights_only=False, mmap=True
        )
        saved = checkpoint["args"]
        saved = vars(saved) if isinstance(saved, argparse.Namespace) else saved
        if saved.get("dataset") != "cifar10" or saved.get("method") not in (
            "fm",
            "fm_ccr_periodic",
        ):
            parser.error("Expected an original CIFAR-10 OT-FM/CCS checkpoint")
        cfg.update({key: saved[key] for key in cfg if key in saved})
        cfg.update(
            epochs=int(checkpoint["epoch"]) + 2,
            resume=str(args.checkpoint.resolve()),
            eval_only=True,
            compute_fid=True,
            eval_frequency=1,
            ode_method="heun2",
            ode_options={"nfe": 100},
            eval_per_class=5000,
            save_fid_samples=True,
            cas_classifier=cas,
        )
        del checkpoint
    elif args.checkpoint:
        parser.error("This label pipeline starts fresh; --checkpoint is for evaluation")
    cfg.update(
        data_path=str(args.data_root.resolve()),
        data_root="",
        output_dir=str(args.output_dir.resolve()),
        num_workers=args.workers,
        dist_on_itp=False,
        dist_url="env://",
        world_size=1,
        start_epoch=0,
    )
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "resolved_config.json").write_text(
        json.dumps(cfg, indent=2) + "\n", encoding="utf8"
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in (
            "torch",
            "torchvision",
            "numpy",
            "scipy",
            "torchmetrics",
            "torch-fidelity",
            "torchdiffeq",
        )
    }
    versions.update(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda)
    (args.output_dir / "environment.json").write_text(
        json.dumps(versions, indent=2) + "\n", encoding="utf8"
    )
    from .engine import main as run_engine

    try:
        run_engine(argparse.Namespace(**cfg))
    finally:
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
    if args.command == "evaluate":
        metrics = json.loads((args.output_dir / "metrics.json").read_text())
        records = [
            json.loads(line)
            for line in (args.output_dir / "log.txt").read_text().splitlines()
            if line.strip()
        ]
        if len(metrics["per_class"]) != 10:
            raise RuntimeError("Final evaluation did not produce all ten classes")
        values = [
            records[-1]["eval_fid"],
            records[-1]["eval_is_mean"],
            metrics["overall"]["cwfid_mean"],
            metrics["overall"]["cw_coverage"],
        ]
        for entry in metrics["per_class"].values():
            values.extend([entry["cwfid"], entry["coverage"]])
        if cas:
            values.append(metrics["overall"]["cas"])
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("Final evaluation contains invalid metrics")
    (args.output_dir / "completed.json").write_text(
        json.dumps({"phase": args.command}) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    main()
