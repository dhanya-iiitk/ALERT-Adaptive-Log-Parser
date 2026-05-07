"""
clustering/adaptive_cluster.py
========================
Step 7: Adaptive agglomerative clustering with entropy-based threshold control.
"""

import math
import numpy as np
from collections import Counter, deque
from typing import Optional

try:
    from sklearn.cluster import AgglomerativeClustering
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


def cluster_logs(
    embeddings:         np.ndarray,
    distance_threshold: float = 0.20,
    linkage:            str   = "average",
) -> np.ndarray:
    """
    Step 7: Run agglomerative clustering.

    Uses two-stage approach:
      Stage 1 — First-keyword partitioning (if log strings available)
      Stage 2 — BERT distance clustering within each group

    For the offline phase, we run directly on embeddings with candidate
    threshold search to pick the best non-degenerate solution.

    Parameters
    ----------
    embeddings          : (N, 768) L2-normalised vectors
    distance_threshold  : τ_d from Module 3
    linkage             : 'average' or 'complete'

    Returns
    -------
    labels : (N,) cluster assignments
    """
    if not SKLEARN_AVAILABLE:
        raise ImportError("scikit-learn required.")

    n    = len(embeddings)
    base = distance_threshold

    # Try candidate thresholds
    candidates = sorted(set([
        max(0.06, base * 0.50),
        max(0.08, base * 0.70),
        max(0.10, base * 0.85),
        base,
        min(0.50, base * 1.12),
    ]))

    results = []
    for thr in candidates:
        lbl   = _run_agg(embeddings, thr, linkage)
        score = _score(lbl, n)
        results.append((score, len(set(lbl)), lbl))

    max_score = max(r[0] for r in results)
    if max_score <= -999:
        best_labels = max(results, key=lambda r: r[1])[2]
    else:
        best_labels = max(results, key=lambda r: r[0])[2]

    n_cl  = len(set(best_labels))
    sizes = Counter(best_labels)
    print(f"[Clustering] {n_cl} clusters  max={max(sizes.values())}  "
          f"avg={np.mean(list(sizes.values())):.1f}")
    return best_labels


def _run_agg(embeddings, threshold, linkage):
    return AgglomerativeClustering(
        n_clusters=None, distance_threshold=threshold,
        metric="euclidean", linkage=linkage,
    ).fit_predict(embeddings)


def _score(labels, n):
    sizes    = list(Counter(labels).values())
    max_frac = max(sizes) / n
    if max_frac > 0.60:
        return -1000.0
    sing_frac = sum(1 for s in sizes if s == 1) / len(sizes)
    return np.log1p(len(sizes)) - sing_frac * 2.0 - max_frac * 5.0


class EntropyThresholdController:
    """
    Module 3: Sliding-window entropy-based threshold adjustment.
    Used during online phase to adapt τ_d.
    """

    def __init__(self, base: float = 0.20, window: int = 500,
                 low: float = 0.30, high: float = 0.80, step: float = 0.01):
        self.threshold = base
        self.min_t     = 0.05
        self.max_t     = 0.55
        self.step      = step
        self.low       = low
        self.high      = high
        self._window   = deque(maxlen=window)

    def record(self, cluster_id: int):
        self._window.append(cluster_id)

    def update(self) -> float:
        if len(self._window) < 10:
            return self.threshold
        counts  = Counter(self._window)
        total   = sum(counts.values())
        entropy = -sum((c/total) * math.log2(c/total) for c in counts.values() if c > 0)
        max_e   = math.log2(max(len(counts), 1))
        norm_e  = entropy / max_e if max_e > 0 else 0.0

        if norm_e > self.high:
            self.threshold = min(self.max_t, self.threshold + self.step)
        elif norm_e < self.low:
            self.threshold = max(self.min_t, self.threshold - self.step)
        return self.threshold
