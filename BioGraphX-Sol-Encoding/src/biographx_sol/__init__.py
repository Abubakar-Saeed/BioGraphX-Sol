"""
BioGraphX-Sol encoding package.

Physicochemical constraint-graph encoder that converts a protein sequence
into a 200-dimensional, per-protein biophysical feature vector for
solubility prediction. All edge/distance reasoning is sequence-based
(linear residue separation) - there is no dependency on experimental or
predicted 3D structure.

Modules
-------
biophysics.py               BioPhysicsStrategy       - interaction rules & AA property scales
graph_engine.py              GraphEngine              - residue interaction graph + graph-topology features
frustration_analyzer.py      FrustrationAnalyzer      - configurational frustration per residue
conservation_proxy.py        ConservationProxy        - alignment-free local-entropy conservation proxy
interface_profiler.py        InteractionMotifScanner,
                              InterfacePropensityEstimator - PPI motif & interface-propensity signal
residue_feature_extractor.py IDPFeatureExtractor      - 60-column per-residue feature stack
solubility_encoder.py        SolubilityEncoder,
                              SolubilityEncoderBatch,
                              run_solubility_pipeline  - aggregates per-residue features into the
                                                         200 per-protein features and drives the
                                                         CSV -> CSV pipeline
utils/feature_names.py       SOLUBILITY_FEATURE_NAMES - the 200 output column names, in order
"""

from .biophysics import BioPhysicsStrategy
from .graph_engine import GraphEngine
from .frustration_analyzer import FrustrationAnalyzer
from .conservation_proxy import ConservationProxy
from .interface_profiler import InteractionMotifScanner, InterfacePropensityEstimator
from .residue_feature_extractor import IDPFeatureExtractor, N_PER_RESIDUE
from .solubility_encoder import SolubilityEncoder, SolubilityEncoderBatch, run_solubility_pipeline
from .utils.feature_names import SELECTED_FEATURE_NAMES, SOLUBILITY_FEATURE_NAMES

__all__ = [
    "BioPhysicsStrategy",
    "GraphEngine",
    "FrustrationAnalyzer",
    "ConservationProxy",
    "InteractionMotifScanner",
    "InterfacePropensityEstimator",
    "IDPFeatureExtractor",
    "N_PER_RESIDUE",
    "SolubilityEncoder",
    "SolubilityEncoderBatch",
    "run_solubility_pipeline",
    "SELECTED_FEATURE_NAMES",
    "SOLUBILITY_FEATURE_NAMES",
]
