# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved. Modifications: Robust Flow Matching authors.
# See licenses/CC-BY-NC-4.0.txt and LICENSE.
# Retained from the successful original CIFAR-10 label-noise pipeline;
# unrelated dataset/method branches were removed without changing its operations.
import gc
import json
import logging
import os
from argparse import Namespace
from pathlib import Path
from typing import Iterable
import PIL.Image
import numpy as np
import torch
import torch.nn.functional as F
from robust_fm.image.label.flow_matching.solver import ODESolver
from robust_fm.image.label.flow_matching.utils import ModelWrapper
from torch.nn.modules import Module
from torch.nn.parallel import DistributedDataParallel
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision.models import inception_v3, Inception_V3_Weights
from torchvision.utils import save_image
from robust_fm.image.label.training import distributed_mode
from robust_fm.image.edm_time_discretization import get_time_discretization

logger = logging.getLogger(__name__)
PRINT_FREQUENCY = 50
CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def sp_artifact_score(
    images: torch.Tensor, lo: float = 0.02, hi: float = 0.98
) -> float:
    """Fraction of pixels very close to 0 or 1 (salt-and-pepper artifact indicator).

    Args:
        images: Tensor in [0, 1] range, shape (N, C, H, W).
        lo: Threshold below which a pixel counts as pepper-like.
        hi: Threshold above which a pixel counts as salt-like.

    Returns:
        Fraction of extreme pixels across the batch (lower is better).
    """
    extreme = (images < lo) | (images > hi)
    return extreme.float().mean().item()


class InceptionFeatureExtractor:
    """Extract pool3 features (2048-d) from InceptionV3 for kNN metrics."""

    def __init__(self, device: torch.device):
        self.device = device
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.transform = weights.transforms()

    @torch.no_grad()
    def extract(self, images: torch.Tensor, batch_size: int = 64) -> np.ndarray:
        """images: (N, 3, H, W) in [0, 1].  Returns (N, 2048) numpy.

        Processes images in chunks of ``batch_size`` to avoid OOM on large N
        (e.g. per-class CW-coverage with 5000 images at 299x299).
        """
        mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
        feats_list = []
        for i in range(0, images.shape[0], batch_size):
            chunk = images[i : i + batch_size].to(self.device, non_blocking=True)
            chunk = F.interpolate(
                chunk, size=(299, 299), mode="bilinear", align_corners=False
            )
            chunk = (chunk - mean) / std
            f = self.model(chunk)
            if isinstance(f, tuple):
                f = f[0]
            feats_list.append(f.cpu().numpy())
            del chunk, f
        return np.concatenate(feats_list, axis=0)


def knn_coverage(
    real_features: np.ndarray, fake_features: np.ndarray, k: int = 5
) -> float:
    """PRDC Coverage (Naeem et al. 2020): fraction of real samples whose
    k-th-NN hypersphere (radius computed among other reals, excluding self)
    contains at least one fake sample.

    Reference impl: https://github.com/clovaai/generative-evaluation-prdc
    """
    from sklearn.metrics import pairwise_distances

    rr = pairwise_distances(real_features, real_features, n_jobs=-1)
    np.fill_diagonal(rr, np.inf)
    real_radii = np.partition(rr, k - 1, axis=1)[:, k - 1]
    rf = pairwise_distances(real_features, fake_features, n_jobs=-1)
    min_to_fake = rf.min(axis=1)
    return float((min_to_fake <= real_radii).mean())


def knn_density(
    real_features: np.ndarray, fake_features: np.ndarray, k: int = 5
) -> float:
    """PRDC Density (Naeem et al. 2020): (1/k) * mean over fakes of the count
    of real k-NN balls containing each fake. Range [0, +inf), 1.0 ~ ideal.

    Reference impl: https://github.com/clovaai/generative-evaluation-prdc
    """
    from sklearn.metrics import pairwise_distances

    rr = pairwise_distances(real_features, real_features, n_jobs=-1)
    np.fill_diagonal(rr, np.inf)
    real_radii = np.partition(rr, k - 1, axis=1)[:, k - 1]
    rf = pairwise_distances(real_features, fake_features, n_jobs=-1)
    inside = rf <= real_radii[:, None]
    counts_per_fake = inside.sum(axis=0)
    return float(counts_per_fake.mean()) / float(k)


class CFGScaledModel(ModelWrapper):
    def __init__(self, model: Module):
        super().__init__(model)
        self.nfe_counter = 0

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cfg_scale: float,
        label: torch.Tensor,
        pure_conditional: bool = False,
    ):
        self.model.module if isinstance(
            self.model, DistributedDataParallel
        ) else self.model
        is_discrete = False
        assert cfg_scale == 0.0 or pure_conditional or (not is_discrete), (
            f"Cfg scaling does not work for the logit outputs of discrete models. Got cfg weight={cfg_scale} and model {type(self.model)}."
        )
        t = torch.zeros(x.shape[0], device=x.device) + t
        with torch.cuda.amp.autocast(), torch.no_grad():
            result = self.model(x, t, extra={"label": label})
        self.nfe_counter += 1
        return result.to(dtype=torch.float32)

    def reset_nfe_counter(self) -> None:
        self.nfe_counter = 0

    def get_nfe(self) -> int:
        return self.nfe_counter


def eval_model(
    model: DistributedDataParallel,
    data_loader: Iterable,
    device: torch.device,
    epoch: int,
    fid_samples: int,
    args: Namespace,
):
    gc.collect()
    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)
    solver = ODESolver(velocity_model=cfg_scaled_model)
    ode_opts = args.ode_options
    fid_metric = FrechetInceptionDistance(normalize=True).to(
        device=device, non_blocking=True
    )
    is_metric = InceptionScore(normalize=True).to(device=device, non_blocking=True)
    feat_extractor = InceptionFeatureExtractor(device)
    real_features_list = []
    fake_features_list = []
    getattr(args, "eval_class", None)
    num_synthetic = 0
    snapshots_saved = False
    artifact_total = 0.0
    artifact_count = 0
    save_images = args.save_fid_samples
    if args.output_dir:
        (Path(args.output_dir) / "snapshots").mkdir(parents=True, exist_ok=True)
    for data_iter_step, batch in enumerate(data_loader):
        samples, labels = (batch[0], batch[1])
        samples = samples.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        fid_metric.update(samples, real=True)
        real_features_list.append(feat_extractor.extract(samples))
        if num_synthetic < fid_samples:
            cfg_scaled_model.reset_nfe_counter()
            gen_labels = labels
            x_0 = torch.randn(samples.shape, dtype=torch.float32, device=device)
            pure_cond = getattr(args, "pure_conditional", False)
            if args.edm_schedule:
                time_grid = get_time_discretization(nfes=ode_opts["nfe"])
            else:
                time_grid = torch.tensor([0.0, 1.0], device=device)
            synthetic_samples = solver.sample(
                time_grid=time_grid,
                x_init=x_0,
                method=args.ode_method,
                return_intermediates=False,
                atol=ode_opts["atol"] if "atol" in ode_opts else 1e-05,
                rtol=ode_opts["rtol"] if "atol" in ode_opts else 1e-05,
                step_size=ode_opts["step_size"] if "step_size" in ode_opts else None,
                label=gen_labels,
                cfg_scale=args.cfg_scale,
                pure_conditional=pure_cond,
            )
            synthetic_samples = torch.clamp(
                synthetic_samples * 0.5 + 0.5, min=0.0, max=1.0
            )
            synthetic_samples = torch.floor(synthetic_samples * 255)
            synthetic_samples = synthetic_samples.to(torch.float32) / 255.0
            logger.info(
                f"{samples.shape[0]} samples generated in {cfg_scaled_model.get_nfe()} evaluations."
            )
            if num_synthetic + synthetic_samples.shape[0] > fid_samples:
                synthetic_samples = synthetic_samples[: fid_samples - num_synthetic]
            fid_metric.update(synthetic_samples, real=False)
            is_metric.update(synthetic_samples)
            fake_features_list.append(feat_extractor.extract(synthetic_samples))
            artifact_total += (
                sp_artifact_score(synthetic_samples) * synthetic_samples.shape[0]
            )
            artifact_count += synthetic_samples.shape[0]
            num_synthetic += synthetic_samples.shape[0]
            if not snapshots_saved and args.output_dir:
                save_image(
                    synthetic_samples,
                    fp=Path(args.output_dir)
                    / "snapshots"
                    / f"{epoch}_{data_iter_step}.png",
                )
                snapshots_saved = True
            if save_images and args.output_dir:
                images_np = (
                    (synthetic_samples * 255.0)
                    .clip(0, 255)
                    .to(torch.uint8)
                    .permute(0, 2, 3, 1)
                    .cpu()
                    .numpy()
                )
                for batch_index, image_np in enumerate(images_np):
                    image_dir = Path(args.output_dir) / "generated"
                    os.makedirs(image_dir, exist_ok=True)
                    image_path = (
                        image_dir
                        / f"{distributed_mode.get_rank()}_{data_iter_step}_{batch_index}.png"
                    )
                    PIL.Image.fromarray(image_np, "RGB").save(image_path)
        if not args.compute_fid:
            return {}
        if data_iter_step % PRINT_FREQUENCY == 0:
            gc.collect()
            running_fid = fid_metric.compute()
            logger.info(
                f"Evaluating [{data_iter_step}/{len(data_loader)}] samples generated [{num_synthetic}/{fid_samples}] running fid {running_fid}"
            )
    stats = {"fid": float(fid_metric.compute().detach().cpu())}
    if artifact_count > 0:
        stats["artifact_score"] = artifact_total / artifact_count
    try:
        is_mean, is_std = is_metric.compute()
        stats["is_mean"] = float(is_mean)
        stats["is_std"] = float(is_std)
    except Exception as e:
        logger.warning(f"IS computation failed: {e}")
    try:
        if real_features_list and fake_features_list:
            all_real = np.concatenate(real_features_list, axis=0)
            all_fake = np.concatenate(fake_features_list, axis=0)
            rng = np.random.RandomState(42)
            if len(all_real) > 10000:
                all_real = all_real[rng.choice(len(all_real), 10000, replace=False)]
            if len(all_fake) > 10000:
                all_fake = all_fake[rng.choice(len(all_fake), 10000, replace=False)]
            stats["coverage"] = knn_coverage(all_real, all_fake, k=5)
            stats["density"] = knn_density(all_real, all_fake, k=5)
    except Exception as e:
        logger.warning(f"Coverage/Density computation failed: {e}")
    del feat_extractor, real_features_list, fake_features_list
    gc.collect()
    return stats


def _generate_class_batch(
    cfg_scaled_model: CFGScaledModel,
    solver: ODESolver,
    batch_size: int,
    class_id: int,
    device: torch.device,
    args: Namespace,
) -> torch.Tensor:
    """Generate a single batch of images for one class. Returns [B,3,H,W] in [0,1]."""
    ode_opts = args.ode_options
    _res = getattr(args, "image_size", 0) or 32
    x_0 = torch.randn(batch_size, 3, _res, _res, dtype=torch.float32, device=device)
    labels = torch.full((batch_size,), class_id, dtype=torch.long, device=device)
    cfg_scaled_model.reset_nfe_counter()
    pure_cond = getattr(args, "pure_conditional", False)
    with torch.no_grad():
        getattr(args, "method", "fm")
        getattr(args, "alpha", 1.0)
        if args.edm_schedule:
            time_grid = get_time_discretization(nfes=ode_opts["nfe"])
        else:
            time_grid = torch.tensor([0.0, 1.0], device=device)
        samples = solver.sample(
            time_grid=time_grid,
            x_init=x_0,
            method=args.ode_method,
            return_intermediates=False,
            atol=ode_opts.get("atol", 1e-05),
            rtol=ode_opts.get("rtol", 1e-05),
            step_size=ode_opts.get("step_size", None),
            label=labels,
            cfg_scale=args.cfg_scale,
            pure_conditional=pure_cond,
        )
    samples = torch.clamp(samples * 0.5 + 0.5, min=0.0, max=1.0)
    samples = torch.floor(samples * 255).to(torch.float32) / 255.0
    return samples


def eval_per_class_metrics(
    model, data_loader: Iterable, device: torch.device, epoch: int, args: Namespace
):
    """
    Per-class evaluation: CAS (Classification Accuracy Score) and CW-FID
    (class-wise FID).

    Only runs when args.eval_per_class > 0.

    Returns dict with per-class and overall metrics, also saves metrics.json.
    """
    num_classes = args.num_classes or 10
    samples_per_class = args.eval_per_class
    batch_size = args.batch_size
    gc.collect()
    cfg_scaled_model = CFGScaledModel(model=model)
    cfg_scaled_model.train(False)
    solver = ODESolver(velocity_model=cfg_scaled_model)
    cas_classifier = None
    cas_normalize = None
    cas_path = getattr(args, "cas_classifier", "")
    if cas_path and os.path.isfile(cas_path):
        from robust_fm.image.label.training.cas_utils import (
            load_classifier,
            CIFAR10_MEAN,
            CIFAR10_STD,
        )

        mean, std = (CIFAR10_MEAN, CIFAR10_STD)
        cas_classifier, cas_normalize = load_classifier(
            cas_path,
            device,
            num_classes=num_classes,
            normalize_mean=mean,
            normalize_std=std,
        )
    else:
        logger.warning(
            f"CAS classifier not found at '{cas_path}'. CAS will be skipped."
        )
    logger.info("Collecting real images per class for CW-FID...")
    real_images_per_class = {c: [] for c in range(num_classes)}
    for batch in data_loader:
        samples, labels = (batch[0], batch[1])
        for c in range(num_classes):
            mask = labels == c
            if mask.any():
                real_images_per_class[c].append(samples[mask])
    for c in range(num_classes):
        if real_images_per_class[c]:
            real_images_per_class[c] = torch.cat(real_images_per_class[c], dim=0)
        else:
            real_images_per_class[c] = torch.empty(
                0,
                3,
                getattr(args, "image_size", 0) or 32,
                getattr(args, "image_size", 0) or 32,
            )
    feat_extractor = InceptionFeatureExtractor(device)
    results = {"per_class": {}, "overall": {}, "epoch": epoch}
    cas_correct_total = 0
    cas_count_total = 0
    cwfid_values = []
    cw_coverage_values = []
    for class_id in range(num_classes):
        cls_name = (
            CIFAR10_CLASSES[class_id]
            if class_id < len(CIFAR10_CLASSES)
            else str(class_id)
        )
        logger.info(
            f"Generating {samples_per_class} samples for class {class_id} ({cls_name})"
        )
        gen_samples = []
        remaining = samples_per_class
        while remaining > 0:
            bs = min(batch_size, remaining)
            batch = _generate_class_batch(
                cfg_scaled_model, solver, bs, class_id, device, args
            )
            gen_samples.append(batch.cpu())
            remaining -= bs
        gen_samples = torch.cat(gen_samples, dim=0)[:samples_per_class]
        a_score = sp_artifact_score(gen_samples)
        cas_acc = None
        if cas_classifier is not None:
            from robust_fm.image.label.training.cas_utils import compute_cas_batch

            correct = 0
            total = 0
            for i in range(0, gen_samples.shape[0], batch_size):
                batch = gen_samples[i : i + batch_size].to(device)
                labels_batch = torch.full(
                    (batch.shape[0],), class_id, dtype=torch.long, device=device
                )
                c, t = compute_cas_batch(
                    batch, labels_batch, cas_classifier, normalize=cas_normalize
                )
                correct += c
                total += t
            cas_acc = correct / total if total > 0 else 0.0
            cas_correct_total += correct
            cas_count_total += total
        cwfid = None
        real_cls = real_images_per_class[class_id]
        if real_cls.shape[0] >= 2 and gen_samples.shape[0] >= 2:
            fid_cls = FrechetInceptionDistance(normalize=True).to(device)
            for i in range(0, real_cls.shape[0], batch_size):
                fid_cls.update(real_cls[i : i + batch_size].to(device), real=True)
            for i in range(0, gen_samples.shape[0], batch_size):
                fid_cls.update(gen_samples[i : i + batch_size].to(device), real=False)
            cwfid = float(fid_cls.compute().detach().cpu())
            cwfid_values.append(cwfid)
            del fid_cls
        cw_cov = None
        if real_cls.shape[0] >= 5 and gen_samples.shape[0] >= 5:
            real_feats_cls = feat_extractor.extract(real_cls.to(device))
            fake_feats_cls = feat_extractor.extract(gen_samples.to(device))
            cw_cov = knn_coverage(real_feats_cls, fake_feats_cls, k=5)
            cw_coverage_values.append(cw_cov)
            del real_feats_cls, fake_feats_cls
        results["per_class"][cls_name] = {
            "class_id": class_id,
            "artifact_score": a_score,
            "num_samples": gen_samples.shape[0],
        }
        if cas_acc is not None:
            results["per_class"][cls_name]["cas"] = cas_acc
        if cwfid is not None:
            results["per_class"][cls_name]["cwfid"] = cwfid
        if cw_cov is not None:
            results["per_class"][cls_name]["coverage"] = cw_cov
        log_parts = [f"  {cls_name}: artifact={a_score:.4f}"]
        if cas_acc is not None:
            log_parts.append(f"CAS={cas_acc:.4f}")
        if cwfid is not None:
            log_parts.append(f"CW-FID={cwfid:.2f}")
        if cw_cov is not None:
            log_parts.append(f"Coverage={cw_cov:.4f}")
        logger.info(", ".join(log_parts))
        if args.output_dir and getattr(args, "save_fid_samples", False):
            grid_dir = Path(args.output_dir) / "per_class_grids"
            grid_dir.mkdir(parents=True, exist_ok=True)
            save_image(
                gen_samples[:64],
                fp=grid_dir / f"epoch{epoch}_class{class_id}_{cls_name}.png",
                nrow=8,
            )
        gc.collect()
    if cas_count_total > 0:
        results["overall"]["cas"] = cas_correct_total / cas_count_total
    if cwfid_values:
        results["overall"]["cwfid_mean"] = sum(cwfid_values) / len(cwfid_values)
    if cw_coverage_values:
        results["overall"]["cw_coverage"] = sum(cw_coverage_values) / len(
            cw_coverage_values
        )
    del feat_extractor
    logger.info("=" * 60)
    logger.info(
        f"{'Class':<15} {'Artifact':>10} {'CAS':>8} {'CW-FID':>10} {'Coverage':>10}"
    )
    logger.info("-" * 60)
    for cls_name, info in results["per_class"].items():
        cas_str = f"{info.get('cas', float('nan')):.4f}" if "cas" in info else "N/A"
        fid_str = f"{info.get('cwfid', float('nan')):.2f}" if "cwfid" in info else "N/A"
        cov_str = (
            f"{info.get('coverage', float('nan')):.4f}" if "coverage" in info else "N/A"
        )
        logger.info(
            f"{cls_name:<15} {info['artifact_score']:>10.4f} {cas_str:>8} {fid_str:>10} {cov_str:>10}"
        )
    logger.info("-" * 60)
    if "cas" in results["overall"]:
        logger.info(f"{'OVERALL CAS':<15} {results['overall']['cas']:>30.4f}")
    if "cwfid_mean" in results["overall"]:
        logger.info(f"{'MEAN CW-FID':<15} {results['overall']['cwfid_mean']:>30.2f}")
    if "cw_coverage" in results["overall"]:
        logger.info(f"{'MEAN CW-COV':<15} {results['overall']['cw_coverage']:>30.4f}")
    logger.info("=" * 60)
    if args.output_dir and distributed_mode.is_main_process():
        metrics_path = Path(args.output_dir) / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    return results
