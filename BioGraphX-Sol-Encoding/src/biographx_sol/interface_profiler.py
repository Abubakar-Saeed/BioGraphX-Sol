"""
InteractionMotifScanner / InterfacePropensityEstimator
================================================================================

Author: Abubakar Saeed

Description:
    InteractionMotifScanner - regex scan for known short linear interaction
    motifs (SH3/WW/PDZ/SH2/14-3-3/Clathrin-box/NR-box/TRAF/ITAM-ITIM/kinase
    docking ligands), flagging every residue they cover.

    InterfacePropensityEstimator - per-residue interface-propensity score
    combining graph surface exposure proxy (betweenness/closeness), scaled
    hydrophobicity, charge, motif flags and local sequence complexity into a
    sigmoid-squashed probability; also returns the surface-exposure proxy
    itself, used elsewhere to mark surface-exposed residues.
"""

import re

import numpy as np
import igraph as ig

from .conservation_proxy import ConservationProxy


class InteractionMotifScanner:
    def __init__(self):
        self.motifs = [
            (r'P.P', 'SH3_ligand'),
            (r'PP.Y', 'WW_ligand'),
            (r'[ST].[VIL]$', 'PDZ_ligand'),
            (r'Y..[VILP]', 'SH2_ligand'),
            (r'R[ST].[ST].P', '14-3-3_ligand'),
            (r'L[LIVMF].[LIVMF][DE]', 'Clathrin_box'),
            (r'[RK].[RK]', 'NR_box'),
            (r'[PS].[QED]', 'TRAF_binding'),
            (r'Y.[LIV].{6,8}Y.[LIV]', 'ITAM_ITIM'),
            (r'[KR]{1,2}.{1,3}[LIVF]', 'Kinase_docking')
        ]

    def get_per_residue_motif_flags(self, sequence: str) -> np.ndarray:
        L = len(sequence)
        flags = np.zeros(L, dtype=bool)
        for pattern, _ in self.motifs:
            for match in re.finditer(pattern, sequence, re.IGNORECASE):
                flags[match.start():match.end()] = True
        return flags


class InterfacePropensityEstimator:
    def __init__(self, biophysics, conservation_proxy: ConservationProxy, motif_scanner: InteractionMotifScanner):
        self.biophys = biophysics
        self.cons_proxy = conservation_proxy
        self.motif_scanner = motif_scanner

    def compute_per_residue(self, graph: ig.Graph, sequence: str):
        L = len(sequence)
        between = np.nan_to_num(np.array(graph.betweenness()), nan=0.0)
        close   = np.nan_to_num(np.array(graph.closeness()),   nan=0.0)
        between_norm = between / (between.max() + 1e-10)
        close_norm = close / (close.max() + 1e-10)
        surface_proxy = 1.0 / (1.0 + np.exp(-(1.0 - 2.0 * between_norm + close_norm)))

        hyd = np.array([self.biophys.hydrophobicity.get(aa, 0.0) for aa in sequence])
        hyd_min, hyd_max = -4.5, 4.5
        hyd_scaled = (hyd - hyd_min) / (hyd_max - hyd_min + 1e-10)

        charge = np.array([1.0 if aa in 'KRDE' else (0.1 if aa == 'H' else 0.0) for aa in sequence])
        motif_flags = self.motif_scanner.get_per_residue_motif_flags(sequence).astype(float)
        cons = self.cons_proxy.per_residue_entropy(sequence)

        raw = 0.3 * surface_proxy + 0.25 * hyd_scaled + 0.15 * charge + 0.2 * motif_flags + 0.1 * cons
        p = 1.0 / (1.0 + np.exp(-raw))
        return p, surface_proxy
