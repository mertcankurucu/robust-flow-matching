"""Single-GPU CIFAR-10 core experiments with explicit outputs and resumable state."""

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, DistributedSampler

from .data import training_dataset
from .model import create_model
from .train import make_state, train_epoch


@dataclass
class Config:
    mode: str = "target"
    method: str = "ot_fm"
    data_root: str = "data"
    device: str = "cuda"
    seed: int = 0
    noise_seed: int = 42
    rate: float = 0.2
    corruption: str = "gaussian"
    sigma: float = 0.3
    noise_type: str = "symmetric"
    epochs: int = 1000
    batch_size: int = 64
    score_batch_size: int = 2048
    workers: int = 8
    lr: float = 1e-4
    tau: float = 0.01
    trs_quantile: float = 0.8  # Fixed across corruption rates; saved for provenance.
    smoke: bool = False

    def validate(self):
        if self.method not in {
            "target": ("ot_fm", "fm_trs"),
            
        }.get(self.mode, ()):
            raise ValueError(
                "This entry point supports target OT-FM/TRS. Use robust_fm.image.label.run for label noise"
            )
        if (
            min(self.epochs, self.batch_size, self.score_batch_size) < 1
            or self.workers < 0
        ):
            raise ValueError("Invalid epoch/batch/worker count")
        if not 0 <= self.rate < 1 or self.sigma < 0 or self.tau <= 0 or self.lr <= 0:
            raise ValueError(
                "Invalid corruption rate, sigma, temperature or learning rate"
            )
        if not 0 < self.trs_quantile < 1:
            raise ValueError("TRS quantile must be in (0,1)")
        if self.smoke:
            self.epochs, self.batch_size, self.score_batch_size, self.workers = (
                4,
                10,
                10,
                0,
            )


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def save_checkpoint(path, model, optimizer, scaler, state, cfg, epoch):
    # Save online parameters, EMA shadows, robust state, and RNG at an epoch boundary.
    if not model.training:
        raise ValueError("Checkpoint must be saved with online weights active")
    np_state = np.random.get_state()
    payload = {
        "format_version": 1,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "state": state,
        "config": asdict(cfg),
        "epoch": epoch,
        "rng_torch": torch.get_rng_state(),
        "rng_python": random.getstate(),
        "rng_numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        "rng_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def recorded_config(config):
    """Recover pre-0.2.2 TRS metadata without relabeling old weights as fixed-80."""
    config = dict(config)
    if "trs_quantile" not in config:
        config["trs_quantile"] = (
            max(0.3, 1 - 1.1 * config["rate"])
            if config["method"] == "fm_trs"
            else 0.8  # Unused for OT-FM.
        )
    return config


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError(
            "Expected a robust-fm-core checkpoint (old development checkpoints use a different schema)"
        )
    checkpoint["config"] = recorded_config(checkpoint["config"])
    return checkpoint


def restore_rng(checkpoint):
    torch.set_rng_state(checkpoint["rng_torch"].cpu())
    random.setstate(checkpoint["rng_python"])
    n = checkpoint["rng_numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    if torch.cuda.is_available() and checkpoint["rng_cuda"]:
        torch.cuda.set_rng_state_all([s.cpu() for s in checkpoint["rng_cuda"]])


def train(cfg, output, resume=None, save_every=10, skip_eval=False):
    cfg.validate()
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError(
            "This core runner uses one GPU per experiment; launch independent jobs for seeds/methods"
        )
    output = Path(output)
    if output.exists() and any(output.iterdir()) and resume is None:
        raise ValueError(
            f"Output already contains files: {output}. Use --resume or a new directory"
        )
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(cfg.device)
    seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = device.type == "cuda"
    dataset = training_dataset(cfg)
    # Same seed/epoch sampler as the original single-rank distributed runner.
    sampler = DistributedSampler(dataset, num_replicas=1, rank=0, shuffle=True)
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    model = create_model(cfg.smoke).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95))
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")
    state = make_state(len(dataset), cfg, device)
    start = 0
    if resume:
        checkpoint = load_checkpoint(resume, device)
        if checkpoint.get("evaluation_only"):
            raise ValueError("This checkpoint was exported for evaluation; resume requires the full training checkpoint")
        previous = checkpoint["config"]
        for key, value in asdict(cfg).items():
            if key not in ("device", "data_root", "workers") and previous[key] != value:
                raise ValueError(
                    f"Resume config mismatch for {key}: {previous[key]} != {value}. "
                    "Use a new output directory for a changed training configuration."
                )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        state = checkpoint["state"]
        start = checkpoint["epoch"] + 1
        restore_rng(checkpoint)
    (output / "config.json").write_text(
        json.dumps(asdict(cfg), indent=2), encoding="utf8"
    )
    for epoch in range(start, cfg.epochs):
        sampler.set_epoch(epoch)
        started = time.monotonic()
        loss = train_epoch(model, loader, optimizer, scaler, state, cfg, epoch, device)
        record = {"epoch": epoch, "loss": loss, "seconds": time.monotonic() - started}
        print(json.dumps(record), flush=True)
        with (output / "train.jsonl").open("a", encoding="utf8") as f:
            f.write(json.dumps(record) + "\n")
        if (epoch + 1) % save_every == 0 or epoch == cfg.epochs - 1:
            save_checkpoint(
                output / "checkpoint.pt", model, optimizer, scaler, state, cfg, epoch
            )
    # Deliberately outside the epoch loop: completed checkpoints can still be evaluated.
    if not skip_eval:
        from .evaluate import evaluate

        evaluate(model, cfg, output / "evaluation", device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("train")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--mode", choices=["target"], default="target")
    p.add_argument("--method", choices=["ot_fm", "fm_trs"], default="ot_fm")
    p.add_argument("--data-root", default="data", help="Existing parent of cifar-10-batches-py/; no dataset download")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-seed", type=int, default=42)
    p.add_argument("--rate", type=float)
    p.add_argument(
        "--corruption", choices=["gaussian", "salt_pepper"], default="gaussian"
    )
    p.add_argument("--sigma", type=float, default=0.3)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--tau", type=float, default=0.01)
    p.add_argument("--resume", type=Path)
    p.add_argument("--save-every", type=int, default=10)
    p.add_argument("--skip-eval", action="store_true")
    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--data-root", default="data", help="Existing parent of cifar-10-batches-py/; no dataset download")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--generated", type=int, default=50000)
    p.add_argument(
        "--per-class",
        type=int,
        default=0,
        choices=[0],
        help="Retained for compatibility; target evaluation is unconditional",
    )
    p.add_argument("--reference-split", choices=["train", "test"], default="train")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--ode-steps",
        type=int,
        default=50,
        help="Heun intervals; approximately twice as many vector-field calls",
    )
    args = vars(parser.parse_args())
    command, output = args.pop("command"), args.pop("output_dir")
    if command == "train":
        resume, save_every, skip_eval = (
            args.pop("resume"),
            args.pop("save_every"),
            args.pop("skip_eval"),
        )
        if save_every < 1:
            parser.error("--save-every must be positive")
        if args["rate"] is None:
            args["rate"] = 0.2
        train(Config(**args), output, resume, save_every, skip_eval)
    else:
        if (
            args["generated"] < 10
            or args["per_class"] not in (0,)
            and args["per_class"] < 6
        ):
            parser.error("Use >=10 generated images and >=6 per class (or 0)")
        if args["batch_size"] < 1 or args["ode_steps"] < 2 or args["workers"] < 0:
            parser.error("Invalid batch size, ODE intervals or worker count")
        device = torch.device(args.pop("device"))
        checkpoint = load_checkpoint(args.pop("checkpoint"), device)
        cfg = Config(**checkpoint["config"])
        cfg.device, cfg.data_root, cfg.workers = (
            str(device),
            args.pop("data_root"),
            args.pop("workers"),
        )
        model = create_model(cfg.smoke).to(device)
        model.load_state_dict(checkpoint["model"])
        from .evaluate import evaluate

        evaluate(model, cfg, output, device, **args)


if __name__ == "__main__":
    main()
