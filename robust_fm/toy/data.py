"""Seeded Torch generators; the same implementation runs on CPU and CUDA."""

import math
import torch


def _generate_labeled_moons_gpu(n_samples, noise=0.05, device="cuda"):
    """GPU moons with integer class labels. Class 0=upper, 1=lower."""
    n_out = n_samples // 2
    n_in = n_samples - n_out
    theta_out = torch.rand(n_out, device=device) * math.pi
    theta_in = torch.rand(n_in, device=device) * math.pi
    outer = torch.stack([torch.cos(theta_out), torch.sin(theta_out)], dim=1)
    inner = torch.stack([1 - torch.cos(theta_in), 1 - torch.sin(theta_in) - 0.5], dim=1)
    data = torch.cat([outer, inner], dim=0)
    labels = torch.cat(
        [
            torch.zeros(n_out, dtype=torch.long, device=device),
            torch.ones(n_in, dtype=torch.long, device=device),
        ]
    )
    if noise > 0:
        data = data + torch.randn_like(data) * noise
    perm = torch.randperm(n_samples, device=device)
    return data[perm], labels[perm]


def _generate_2spirals_gpu(n_samples, device="cuda"):
    n_per = n_samples // 2
    n_rest = n_samples - n_per
    t1 = torch.sqrt(torch.rand(n_per, device=device)) * 3 * math.pi
    t2 = torch.sqrt(torch.rand(n_rest, device=device)) * 3 * math.pi
    s1 = torch.stack([t1 * torch.cos(t1) / 3, t1 * torch.sin(t1) / 3], dim=1)
    s2 = torch.stack([-t2 * torch.cos(t2) / 3, -t2 * torch.sin(t2) / 3], dim=1)
    data = torch.cat([s1, s2], dim=0)
    return data[torch.randperm(n_samples, device=device)]


def _generate_labeled_2spirals_gpu(n_samples, device="cuda"):
    """GPU 2-spirals with class labels. Class 0=spiral 1, class 1=spiral 2."""
    n_per = n_samples // 2
    n_rest = n_samples - n_per
    t1 = torch.sqrt(torch.rand(n_per, device=device)) * 3 * math.pi
    t2 = torch.sqrt(torch.rand(n_rest, device=device)) * 3 * math.pi
    s1 = torch.stack([t1 * torch.cos(t1) / 3, t1 * torch.sin(t1) / 3], dim=1)
    s2 = torch.stack([-t2 * torch.cos(t2) / 3, -t2 * torch.sin(t2) / 3], dim=1)
    data = torch.cat([s1, s2], dim=0)
    labels = torch.cat(
        [
            torch.zeros(n_per, dtype=torch.long, device=device),
            torch.ones(n_rest, dtype=torch.long, device=device),
        ]
    )
    perm = torch.randperm(n_samples, device=device)
    return data[perm], labels[perm]


def _generate_labeled_segmented_spirals_gpu(
    n_samples, n_segments=5, noise=0.05, device="cuda"
):
    """GPU segmented spirals with 2*n_segments classes.

    Each spiral (A/B) is split into ``n_segments`` equal arc segments.
    Spiral A segments get classes 0..n_segments-1,
    Spiral B segments get classes n_segments..2*n_segments-1.

    Total classes = 2 * n_segments (default 10).
    """
    n_per = n_samples // 2
    n_rest = n_samples - n_per

    seg_width = 3.0 * math.pi / n_segments  # width of each segment in t

    # Spiral A
    t1 = torch.sqrt(torch.rand(n_per, device=device)) * 3 * math.pi
    s1 = torch.stack([t1 * torch.cos(t1) / 3, t1 * torch.sin(t1) / 3], dim=1)
    seg_a = torch.clamp((t1 / seg_width).long(), 0, n_segments - 1)
    labels_a = seg_a  # classes 0..n_segments-1

    # Spiral B (negated)
    t2 = torch.sqrt(torch.rand(n_rest, device=device)) * 3 * math.pi
    s2 = torch.stack([-t2 * torch.cos(t2) / 3, -t2 * torch.sin(t2) / 3], dim=1)
    seg_b = torch.clamp((t2 / seg_width).long(), 0, n_segments - 1)
    labels_b = seg_b + n_segments  # classes n_segments..2*n_segments-1

    data = torch.cat([s1, s2], dim=0)
    labels = torch.cat([labels_a, labels_b], dim=0)

    if noise > 0:
        data = data + torch.randn_like(data) * noise

    perm = torch.randperm(n_samples, device=device)
    return data[perm], labels[perm]


def inject_label_noise(labels, p_flip, seed=None):
    """
    Inject label flip noise: with probability p_flip, reassign to a random
    different class. Works for any number of classes.

    Args:
        labels: torch.Tensor of shape (N,) with integer class values
        p_flip: float in [0, 1], probability of flipping each label
        seed: optional random seed for reproducibility

    Returns:
        noisy_labels: torch.Tensor with flipped labels
        realized_flip_rate: actual fraction of labels that were flipped
    """
    if p_flip == 0.0:
        return labels.clone(), 0.0

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    n_classes = int(labels.max().item()) + 1
    device = labels.device

    # Sample flip mask (generator only works on CPU, so generate there and move)
    flip_mask = (torch.rand(len(labels), generator=rng) < p_flip).to(device)

    noisy_labels = labels.clone()
    if n_classes == 2:
        # Binary: simple flip
        noisy_labels[flip_mask] = 1 - noisy_labels[flip_mask]
    else:
        # Multi-class: reassign to a uniformly random *different* class
        n_flip = flip_mask.sum().item()
        if n_flip > 0:
            offsets = torch.randint(1, n_classes, (n_flip,), generator=rng).to(device)
            noisy_labels[flip_mask] = (noisy_labels[flip_mask] + offsets) % n_classes

    realized_flip_rate = flip_mask.float().mean().item()

    return noisy_labels, realized_flip_rate


def inject_adjacent_label_noise(labels, p_flip, n_segments=5, seed=None):
    """
    Inject adjacent-segment label noise for segmented spirals.

    With probability p_flip, each label is flipped to an adjacent segment
    on the *same* spiral arm (wrapping within the arm's segment range).

    Classes 0..n_segments-1 are spiral A, n_segments..2*n_segments-1 are spiral B.
    Adjacent means +1 or -1 within the same moon (with wrap-around).

    Args:
        labels: torch.Tensor of shape (N,) with class values in [0, 2*n_segments)
        p_flip: probability of flipping each label
        n_segments: number of segments per arm (default 5)
        seed: optional random seed

    Returns:
        noisy_labels, realized_flip_rate
    """
    if p_flip == 0.0:
        return labels.clone(), 0.0

    if seed is not None:
        rng = torch.Generator()
        rng.manual_seed(seed)
    else:
        rng = None

    device = labels.device
    flip_mask = (torch.rand(len(labels), generator=rng) < p_flip).to(device)
    noisy_labels = labels.clone()

    n_flip = flip_mask.sum().item()
    if n_flip > 0:
        # Determine which moon each sample belongs to
        moon_id = (labels[flip_mask] >= n_segments).long()  # 0=upper, 1=lower
        base = moon_id * n_segments  # 0 for upper, n_segments for lower
        local_seg = (
            labels[flip_mask] - base
        )  # segment index within moon [0, n_segments)

        # Random direction: +1 or -1
        direction = (torch.randint(0, 2, (n_flip,), generator=rng) * 2 - 1).to(device)
        new_local = (local_seg + direction) % n_segments
        noisy_labels[flip_mask] = base + new_local

    realized_flip_rate = flip_mask.float().mean().item()
    return noisy_labels, realized_flip_rate
