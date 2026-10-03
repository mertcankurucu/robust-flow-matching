"""Check existing CIFAR-10 data; optionally cache pretrained metric weights."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--cache-metrics", action="store_true",
        help="Also cache Inception weights (may download missing weights)",
    )
    args = parser.parse_args()
    from torchvision.datasets import CIFAR10
    print(f"Checking existing CIFAR-10: {args.data_root}", flush=True)
    dataset = CIFAR10(root=str(args.data_root.expanduser()), train=True, download=False)
    print(f"Dataset ready: {len(dataset)} training images; no download.", flush=True)
    if not args.cache_metrics:
        return
    from torchvision.models import Inception_V3_Weights, inception_v3
    from torchmetrics.image.fid import FrechetInceptionDistance

    print("Caching FID/Inception Score weights", flush=True)
    fid = FrechetInceptionDistance(normalize=True)
    del fid
    print("Caching coverage feature-extractor weights", flush=True)
    features = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    del features
    print("Metric weights ready.", flush=True)


if __name__ == "__main__":
    main()
