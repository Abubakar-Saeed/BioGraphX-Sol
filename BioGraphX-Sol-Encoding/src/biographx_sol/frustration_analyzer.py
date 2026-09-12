"""
FrustrationAnalyzer
================================================================================

Author: Abubakar Saeed

Description:
    Per-residue configurational frustration: the variance of interaction
    energies (edge weight x interaction strength) over each residue's
    incident edges in the BioGraphX-Sol interaction graph, normalized to
    [0, 1] by the sequence maximum.
"""

import numpy as np
import igraph as ig

from .biophysics import BioPhysicsStrategy


class FrustrationAnalyzer:
    def __init__(self, biophysics_strategy: BioPhysicsStrategy):
        self.biophys = biophysics_strategy

    def compute_frustration(self, graph: ig.Graph) -> np.ndarray:
        n = graph.vcount()
        if graph.ecount() == 0:
            return np.zeros(n, dtype=np.float32)
        frustration = np.zeros(n)
        for pos in range(n):
            inc = graph.incident(pos)
            if not inc:
                continue
            energies = []
            for eid in inc:
                w = graph.es[eid]["weight"]
                itype = graph.es[eid]["interaction_type"]
                s = self.biophys.interaction_rules.get(itype, {}).get("strength", 1.0)
                energies.append(w * s)
            frustration[pos] = np.var(energies) if len(energies)>1 else 0.0
        max_f = frustration.max()
        if max_f > 0:
            frustration /= max_f
        return frustration.astype(np.float32)
