"""
GraphEngine
================================================================================

Author: Abubakar Saeed

Description:
    Builds a residue-interaction graph from a protein sequence using the
    interaction rules in BioPhysicsStrategy (edges are LINEAR SEQUENCE
    SEPARATION-gated, not 3D distance - the whole pipeline is
    sequence-derived with no structural dependency), and extracts
    per-residue topology features (degree, weighted degree, betweenness,
    closeness, clustering, hybrid-interaction ratio) from it.
"""

import numpy as np
import igraph as ig

from .biophysics import BioPhysicsStrategy


class GraphEngine:
    def __init__(self, biophysics_strategy: BioPhysicsStrategy):
        self.biophys = biophysics_strategy

    def build_complete_graph(self, sequence: str) -> ig.Graph:
        n = len(sequence)
        seq = sequence.upper()
        graph = ig.Graph(n, directed=False)
        graph.vs["residue"] = list(seq)
        edges = []
        edge_attrs = {"weight":[], "interaction_type":[], "is_hybrid":[]}
        for i in range(n):
            # Non-disulfide interactions (max distance 35)
            max_j = min(n, i + 36)
            for j in range(i+1, max_j):
                distance = j - i
                aa1, aa2 = seq[i], seq[j]
                found = set()
                for itype, rules in self.biophys.interaction_rules.items():
                    if distance > rules['max_distance']: continue
                    if self.biophys.check_interaction(aa1, aa2, itype):
                        found.add(itype)
                if found:
                    dominant = max(found, key=lambda t: self.biophys.interaction_rules[t]['strength'])
                    w = self.biophys.interaction_rules[dominant]['strength']
                    w *= (1 - (distance / self.biophys.interaction_rules[dominant]['max_distance']) * 0.3)
                    hybrid = 0
                    if len(found) >= 2:
                        for hy, hr in self.biophys.hybrid_interactions.items():
                            if hr['primary'] in found and hr['secondary'] in found:
                                hybrid = 1
                                w *= hr['weight']
                                break
                    edges.append((i, j))
                    edge_attrs["weight"].append(min(w, 1.0))
                    edge_attrs["interaction_type"].append(dominant)
                    edge_attrs["is_hybrid"].append(hybrid)
            # Disulfide check for large distances (max 2000)
            if seq[i] == 'C':
                for j in range(i+36, min(n, i+2001)):
                    if seq[j] == 'C':
                        distance = j - i
                        found = {'disulfide'}
                        dominant = 'disulfide'
                        w = 0.9
                        w *= (1 - (distance / 2000) * 0.3)
                        hybrid = 0
                        edges.append((i, j))
                        edge_attrs["weight"].append(min(w, 1.0))
                        edge_attrs["interaction_type"].append(dominant)
                        edge_attrs["is_hybrid"].append(hybrid)
        if edges:
            graph.add_edges(edges)
            for attr, values in edge_attrs.items():
                graph.es[attr] = values
        backbone = [(i, i+1) for i in range(n-1) if not graph.are_adjacent(i, i+1)]
        if backbone:
            start = graph.ecount()
            graph.add_edges(backbone)
            graph.es[start:]["weight"] = [1.0]*len(backbone)
            graph.es[start:]["interaction_type"] = ["backbone"]*len(backbone)
            graph.es[start:]["is_hybrid"] = [0]*len(backbone)
        return graph

    def extract_per_residue_graph_features(self, graph: ig.Graph):
        n = graph.vcount()
        degree = np.array(graph.degree(), dtype=np.float32)
        weighted_degree = np.array(graph.strength(weights="weight"), dtype=np.float32) if graph.ecount()>0 else np.zeros(n)
        between = np.nan_to_num(np.array(graph.betweenness()), nan=0.0)
        close = np.nan_to_num(np.array(graph.closeness()), nan=0.0)
        clustering = np.nan_to_num(np.array(graph.transitivity_local_undirected(mode="zero")), nan=0.0)
        hybrid_ratio = np.zeros(n, dtype=np.float32)
        if graph.ecount()>0 and "is_hybrid" in graph.es.attributes():
            for v in range(n):
                inc = graph.incident(v)
                if inc:
                    hybrid_ratio[v] = sum(graph.es[eid]["is_hybrid"] for eid in inc) / len(inc)
        return {
            "degree": degree,
            "weighted_degree": weighted_degree,
            "betweenness": between,
            "closeness": close,
            "clustering": clustering,
            "hybrid_ratio": hybrid_ratio
        }
