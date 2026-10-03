"""Exact W2, RBF MMD and memory-bounded PRDC metrics; no fallback proxies."""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors


def wasserstein2(real, fake):
    import ot

    cost = cdist(real, fake, metric="sqeuclidean")
    value = ot.emd2(
        np.ones(len(real)) / len(real),
        np.ones(len(fake)) / len(fake),
        cost,
        numItermax=1000000,
    )
    return float(np.sqrt(max(0, value)))


def mmd(real, fake):
    both = np.concatenate([real, fake])
    distances = cdist(both, both)
    positive = distances[distances > 0]
    bandwidth = np.median(positive) if len(positive) else 1.0
    del distances, positive

    def kernel_mean(a, b):
        total = 0.0
        for start in range(0, len(a), 256):
            d = cdist(a[start : start + 256], b, metric="sqeuclidean")
            total += np.exp(-d / (2 * bandwidth**2)).sum()
        return total / (len(a) * len(b))

    return float(
        np.sqrt(
            max(
                0,
                kernel_mean(real, real)
                + kernel_mean(fake, fake)
                - 2 * kernel_mean(real, fake),
            )
        )
    )


def coverage_density(real, fake, k=5):
    if len(real) <= k or len(fake) == 0:
        raise ValueError(f"PRDC requires >{k} real samples and at least one fake.")
    # Querying the fitted data implicitly excludes each point's own index.
    radii = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(real).kneighbors()[0][:, -1]
    covered, inside = 0, 0
    for start in range(0, len(real), 256):
        distances = pairwise_distances(real[start : start + 256], fake, n_jobs=1)
        hits = distances <= radii[start : start + 256, None]
        covered += np.any(hits, axis=1).sum()
        inside += hits.sum()
    return {
        "coverage": float(covered / len(real)),
        "density": float(inside / (len(fake) * k)),
    }


def toy_metrics(real, fake, max_ot=10000):
    rng = np.random.RandomState(42)

    def subset(x):
        return x if len(x) <= max_ot else x[rng.choice(len(x), max_ot, replace=False)]

    a, b = subset(real), subset(fake)
    scores = {"w2_clean": wasserstein2(a, b), "mmd": mmd(a, b)}
    scores["coverage"] = coverage_density(real, fake)["coverage"]
    scores["density"] = coverage_density(a, b)["density"]
    return scores
