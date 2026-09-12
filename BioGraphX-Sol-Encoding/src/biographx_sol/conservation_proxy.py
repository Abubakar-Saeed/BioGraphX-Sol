"""
ConservationProxy
================================================================================

Author: Abubakar Saeed

Description:
    Alignment-free local-entropy proxy for per-residue sequence complexity:
    Shannon entropy of the amino-acid composition in a sliding window
    (normalized by log(20)), inverted so higher values mean lower local
    complexity / more repetitive context. Used at multiple window radii to
    build the multi-scale complexity features.
"""

from collections import Counter
from math import log

import numpy as np


class ConservationProxy:
    def __init__(self, window_radius=10):
        self.window_radius = window_radius

    def per_residue_entropy(self, sequence: str, window_radius=None) -> np.ndarray:
        if window_radius is None:
            window_radius = self.window_radius
        L = len(sequence)
        entropy = np.zeros(L, dtype=np.float32)
        for i in range(L):
            start = max(0, i - window_radius)
            end = min(L, i + window_radius + 1)
            cnt = Counter(sequence[start:end])
            total = end - start
            ent = -sum((c/total)*log(c/total) for c in cnt.values())
            entropy[i] = ent / log(20)
        return 1.0 - entropy
