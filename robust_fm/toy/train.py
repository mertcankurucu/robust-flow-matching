"""The three core methods, retaining the latest CUDA sweep's training rules."""

import time
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from . import data
from .models import ConditionalVectorField, VectorField
from ..metrics import coverage_density, toy_metrics, wasserstein2


@dataclass
class Config:
    mode: str = "target"
    dataset: str = "2spirals"
    method: str = "ot_fm"
    seed: int = 0
    device: str = "cuda"
    steps: int = 10001
    batch_size: int = 4096
    hidden_dim: int = 256
    pool_size: int = 50000
    generated: int = 50000  # TOTAL samples, including for conditional runs.
    ode_steps: int = 20
    tau: float = 0.001
    rate: float = 0.2
    distance: float = 4.0
    noise_type: str = "symmetric"
    score_freeze_steps: int = 2000
    max_ot: int = 10000

    def validate(self):
        allowed = {"target": ("ot_fm", "fm_trs"), "label": ("ot_fm", "fm_ccs")}
        if self.mode not in allowed or self.method not in allowed[self.mode]:
            raise ValueError(
                "Target mode supports OT-FM/TRS; label mode supports OT-FM/CCS."
            )
        if self.dataset not in ("moons", "2spirals", "segmented_spirals"):
            raise ValueError("Unsupported 2D dataset")
        if (
            min(
                self.steps,
                self.batch_size,
                self.hidden_dim,
                self.pool_size,
                self.generated,
                self.ode_steps,
                self.max_ot,
            )
            < 1
            or self.tau <= 0
        ):
            raise ValueError("Counts and temperature must be positive")
        if not 0 <= self.rate < 1 or self.score_freeze_steps < 0:
            raise ValueError(
                "Noise rate must be in [0,1); freeze steps must be nonnegative"
            )
        if self.noise_type not in ("symmetric", "adjacent"):
            raise ValueError("Unknown label noise type")
        if self.noise_type == "adjacent" and self.dataset != "segmented_spirals":
            raise ValueError("Adjacent-segment noise requires segmented_spirals")
        if self.mode == "label" and self.generated % self.classes:
            raise ValueError(
                "Conditional sample count must be divisible by the class count"
            )

    @property
    def classes(self):
        return 10 if self.dataset == "segmented_spirals" else 2

    @property
    def source_std(self):
        return 2.0 if self.dataset == "segmented_spirals" else 0.05


def generate_data(cfg, count, labeled=True):
    if cfg.dataset == "moons":
        x, labels = data._generate_labeled_moons_gpu(count, 0.05, cfg.device)
        x[:, 1] += torch.where(labels == 0, 0.25, -0.25)
    elif cfg.dataset == "2spirals":
        if not labeled:
            return data._generate_2spirals_gpu(count, cfg.device), None
        x, labels = data._generate_labeled_2spirals_gpu(count, cfg.device)
    else:
        x, labels = data._generate_labeled_segmented_spirals_gpu(
            count, device=cfg.device
        )
    return x, labels


def make_pool(cfg):
    x1, labels = generate_data(cfg, cfg.pool_size, labeled=cfg.mode == "label")
    if cfg.mode == "target" and cfg.rate > 0:
        n_outlier = max(1, int(cfg.pool_size * cfg.rate / (1 - cfg.rate)))
        center = torch.tensor([cfg.distance, 0.0], device=cfg.device)
        outliers = torch.randn(n_outlier, 2, device=cfg.device) * 0.3 + center
        x1 = torch.cat([x1, outliers])
    x0 = torch.randn(len(x1), 2, device=cfg.device) * cfg.source_std
    if cfg.mode == "label":
        inject = (
            data.inject_label_noise
            if cfg.noise_type == "symmetric"
            else data.inject_adjacent_label_noise
        )
        labels, _ = inject(labels, cfg.rate)
    return x0, x1, labels


@torch.no_grad()
def ccs_margins(model, x0, x1, labels, classes):
    # 2D convention: positive means the assigned class has LOWER residual.
    scores = torch.zeros(len(x1), classes, device=x1.device)
    for t_value in (0.2, 0.5, 0.8):
        for start in range(0, len(x1), 4096):
            stop = start + 4096
            target = x1[start:stop] - x0[start:stop]
            xt = (1 - t_value) * x0[start:stop] + t_value * x1[start:stop]
            t = torch.full((len(xt),), t_value, device=x1.device)
            for c in range(classes):
                y = torch.full_like(labels[start:stop], c)
                scores[start:stop, c] += ((model(xt, t, y) - target) ** 2).sum(1)
    scores /= 3
    assigned = scores.gather(1, labels[:, None]).squeeze(1)
    scores.scatter_(1, labels[:, None], float("inf"))
    return scores.min(1).values - assigned


def fit(cfg):
    cfg.validate()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    model = (
        ConditionalVectorField(dim=2, n_classes=cfg.classes, hidden_dim=cfg.hidden_dim)
        if cfg.mode == "label"
        else VectorField(dim=2, hidden_dim=cfg.hidden_dim)
    )
    model = model.to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x0_pool, x1_pool, labels = make_pool(cfg)
    scores = torch.zeros(len(x1_pool), device=cfg.device)
    counts = torch.zeros_like(scores)
    refresh = sorted({int(f * cfg.steps) for f in (0.25, 0.5)})
    model.train()
    started = time.monotonic()
    for step in range(cfg.steps):
        idx = torch.randint(len(x1_pool), (cfg.batch_size,), device=cfg.device)
        x1 = x1_pool[idx]
        x0 = (
            x0_pool[idx]
            if cfg.mode == "target"
            else torch.randn_like(x1) * cfg.source_std
        )
        t = torch.rand(len(idx), device=cfg.device)
        if cfg.method == "fm_trs":
            t = torch.sqrt(t)
        xt = (1 - t[:, None]) * x0 + t[:, None] * x1
        pred = model(xt, t, labels[idx]) if cfg.mode == "label" else model(xt, t)
        per_sample = ((pred - (x1 - x0)) ** 2).mean(1)
        weights = torch.ones_like(per_sample)
        if cfg.method == "fm_trs":
            with torch.no_grad():
                # Retains the original indexed EMA update for repeated pool indices.
                scores[idx] = torch.where(
                    counts[idx] > 0,
                    0.99 * scores[idx] + 0.01 * per_sample.detach(),
                    per_sample.detach(),
                )
                counts[idx] += 1
                if step >= cfg.score_freeze_steps:
                    threshold = torch.quantile(scores[counts > 0], 0.6)
                    weights = torch.sigmoid(-(scores[idx] - threshold) / cfg.tau)
        elif cfg.method == "fm_ccs":
            if step in refresh:
                model.eval()
                x0_sweep = torch.randn_like(x1_pool) * cfg.source_std
                margins = ccs_margins(model, x0_sweep, x1_pool, labels, cfg.classes)
                scores = (
                    margins if step == refresh[0] else 0.99 * scores + 0.01 * margins
                )
                model.train()
            if step >= refresh[0]:
                weights = torch.sigmoid(scores[idx] / cfg.tau)
        loss = (weights * per_sample).mean()
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite training loss at step {step}")
        optimizer.zero_grad()
        loss.backward()
        # Latest 2D sweep starts the TRS score-only phase at step zero.
        if cfg.method != "fm_trs" or step >= cfg.score_freeze_steps:
            optimizer.step()
        if step % 1000 == 0 or step == cfg.steps - 1:
            print(
                f"{cfg.method} {step + 1}/{cfg.steps}: loss={loss.item():.6g}",
                flush=True,
            )
    return (
        model,
        (x1_pool, labels),
        {"final_loss": loss.item(), "train_time_s": time.monotonic() - started},
    )


@torch.no_grad()
def sample(model, cfg):
    model.eval()
    rng = np.random.RandomState(cfg.seed + 999)
    samples, labels = [], []
    for c in range(cfg.classes if cfg.mode == "label" else 1):
        count = cfg.generated // cfg.classes if cfg.mode == "label" else cfg.generated
        source = torch.tensor(
            rng.randn(count, 2).astype(np.float32) * cfg.source_std,
            dtype=torch.float32,
            device=cfg.device,
        )
        for start in range(0, count, 4096):
            x = source[start : start + 4096]
            y = torch.full((len(x),), c, device=cfg.device, dtype=torch.long)
            for i in range(cfg.ode_steps):
                dt = 1.0 / cfg.ode_steps
                t = torch.full((len(x),), i * dt, device=cfg.device)
                velocity = model(x, t, y) if cfg.mode == "label" else model(x, t)
                x = x + dt * velocity
            samples.append(x.cpu().numpy())
            labels.append(y.cpu().numpy())
    return np.concatenate(samples), np.concatenate(labels)


def run(cfg, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output must be new/empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    model, pool, metrics = fit(cfg)
    generated, generated_labels = sample(model, cfg)
    arrays = {"training": pool[0].cpu().numpy(), "generated": generated}
    if cfg.mode == "label":
        arrays.update(
            training_labels=pool[1].cpu().numpy(), generated_labels=generated_labels
        )
    np.savez_compressed(output / "figure_data.npz", **arrays)
    torch.save(
        {"model": model.state_dict(), "config": asdict(cfg)}, output / "model.pt"
    )
    # Independent, reproducible reference; never consumes training RNG.
    torch.manual_seed(cfg.seed + (10000 if cfg.mode == "target" else 20000))
    real, real_labels = generate_data(cfg, 10000)
    real, real_labels = real.cpu().numpy(), real_labels.cpu().numpy()
    if cfg.mode == "target":
        metrics.update(toy_metrics(real, generated, cfg.max_ot))
    else:
        per_class = []
        for c in range(cfg.classes):
            a, b = (
                real[real_labels == c][:5000],
                generated[generated_labels == c][:5000],
            )
            per_class.append(
                {"class": c, "w2": wasserstein2(a, b), **coverage_density(a, b)}
            )
        metrics.update(
            cw_w2=float(np.mean([r["w2"] for r in per_class])),
            cw_coverage=float(np.mean([r["coverage"] for r in per_class])),
            per_class=per_class,
        )
    result = {"config": asdict(cfg), "metrics": metrics}
    (output / "result.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf8"
    )
    return result
