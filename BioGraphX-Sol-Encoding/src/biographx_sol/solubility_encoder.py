"""
SolubilityEncoder / SolubilityEncoderBatch / run_solubility_pipeline
================================================================================

Author: Abubakar Saeed

Description:
    Master orchestration for the BioGraphX-Sol encoder. Runs
    IDPFeatureExtractor's 60-column per-residue features for a sequence,
    then aggregates them into the 200 per-protein features actually used by
    the models: retained mean/std/min/max per per-residue signal, means over
    surface-exposed residues and N-/C-terminal 30-residue windows, 10 global
    sequence features (pI, charge density, aromaticity, instability index,
    aliphatic index, cysteine count, hydrophobic-stretch statistics, mean
    WALTZ/amyloid propensity), sequence charge-decoration (SCD) and
    N/C-terminal charge asymmetry, and 5 aggregation-hotspot features
    (a hydrophobicity + sheet-propensity - |charge| score used to flag
    aggregation-prone stretches).

    `run_solubility_pipeline(input_csv, output_csv, chunk_size, n_jobs)` is
    the high-throughput CSV -> CSV entry point used by `run.py`: reads a
    `sequence` column (and `gene` as the ID, optionally `solubility` as the
    label) and writes one row per protein (`ID, [Label,] <200 features>`).
"""

import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

from .biophysics import BioPhysicsStrategy
from .residue_feature_extractor import IDPFeatureExtractor, N_PER_RESIDUE
from .utils.feature_names import SELECTED_FEATURE_NAMES, SOLUBILITY_FEATURE_NAMES


class SolubilityEncoder:
    _STATS = ('mean', 'std', 'min', 'max')
    _STAT_FN = {'mean': np.mean, 'std': np.std, 'min': np.min, 'max': np.max}

    def __init__(self):
        self.biophys = BioPhysicsStrategy()
        self.idp_extractor = IDPFeatureExtractor()

        # Order here must match the column order produced by
        # IDPFeatureExtractor.extract_features().
        self.per_residue_names = [
            "Degree", "Weighted_Degree", "Betweenness", "Closeness", "Clustering", "Hybrid_Ratio",
            "Frustration", "Sequence_Complexity", "Disorder_AA_Scale",
            "Local_Hydrophobicity", "Local_Charge", "Local_Helix_Propensity", "Local_Sheet_Propensity",
            "Disorder_Res_Fraction", "Order_Res_Fraction",
            "NTerm_Distance", "CTerm_Distance",
            "GRAVY_Global", "Net_Charge_Global", "Length_Normalized",
            "Degree_Log", "Betweenness_Log", "Weighted_Degree_Log", "Closeness_Log",
            "Interface_Propensity", "Disorder_from_Ifprop",
            "In_Motif", "Local_Motif_Density",
            "Hydro_Autocorr_Lag1", "Charge_Autocorr_Lag1",
            "Frustration_Entropy_Win7", "Uversky_Distance", "Complexity_Std_Win7",
            "Global_Disorder_Composition_Bias", "Disorder_AA_Scale_Local", "Surface_Proxy",
            "Local_Edge_Density",
            "Predicted_Helix", "Predicted_Sheet", "Predicted_Coil",
            "OrderDisorder_Pattern_Entropy", "Local_Bulkiness", "Local_Charge_LargeWin",
            "Local_Kappa", "Sticker_Density", "Spacer_Density", "Sticker_Dispersion",
            "ProGly_Density",
            "MoRF_Core_Score", "MoRF_Flank_Score", "MoRF_Dip_Score",
            "FCR", "NCPR", "Polyampholyte_Score",
            "Complexity_W5", "Complexity_W25", "Complexity_W50", "Complexity_Scale_Variance",
            "Contact_Order", "Ramachandran_Flexibility"
        ]
        self.n_per_residue = len(self.per_residue_names)
        if self.n_per_residue != N_PER_RESIDUE:
            raise AssertionError(
                f"per_residue_names has {self.n_per_residue} entries, expected {N_PER_RESIDUE}"
            )
        self._pr_idx = {n: i for i, n in enumerate(self.per_residue_names)}

        self.solubility_key_features = [
            "Local_Hydrophobicity", "Local_Charge", "Disorder_AA_Scale",
            "Local_Sheet_Propensity", "Sticker_Density", "Spacer_Density",
            "MoRF_Core_Score", "Contact_Order", "Frustration",
            "Hybrid_Ratio"
        ]

        # Derive, from the selected-feature list, exactly which statistics /
        # regional summaries actually need to be computed.
        wanted = set(SELECTED_FEATURE_NAMES)
        self._selected_stats = {
            n: [s for s in self._STATS if f"{n}_{s}" in wanted]
            for n in self.per_residue_names
            if any(f"{n}_{s}" in wanted for s in self._STATS)
        }
        self._selected_surf = [f for f in self.solubility_key_features if f"{f}_surf" in wanted]
        self._selected_nterm = [f for f in self.solubility_key_features if f"{f}_Nterm" in wanted]
        self._selected_cterm = [f for f in self.solubility_key_features if f"{f}_Cterm" in wanted]
        self._wanted = wanted

    def _compute_global_features(self, seq: str, hyd: np.ndarray, chg: np.ndarray) -> Dict[str, float]:
        """Only the 10 global features retained in SELECTED_FEATURE_NAMES."""
        L = len(seq)
        feats = {}

        net_chg = np.sum(chg)
        feats['Net_Charge_pH7'] = net_chg
        feats['Charge_Density'] = net_chg / L if L > 0 else 0.0
        feats['pI'] = self.biophys.calculate_isoelectric_point(seq)

        arom = sum(1 for aa in seq if aa in 'FYW')
        feats['Aromaticity'] = arom / L if L > 0 else 0.0

        instability = 0.0
        for i in range(L-1):
            dipep = seq[i:i+2]
            instability += self.biophys.instability_weights.get(dipep, 0.0)
        feats['Instability_Index'] = instability / L if L > 0 else 0.0

        ala = seq.count('A')
        val = seq.count('V')
        ile = seq.count('I')
        leu = seq.count('L')
        feats['Aliphatic_Index'] = (ala + 2.9*val + 3.9*ile + 3.9*leu) / L * 100 if L > 0 else 0.0

        feats['Cysteine_Count'] = seq.count('C')

        hydrophobic_aas = set('AVILMFWY')
        lengths = []          # stretches with length >= 5
        current = 0
        max_stretch = 0       # global maximum, regardless of length
        for aa in seq:
            if aa in hydrophobic_aas:
                current += 1
                max_stretch = max(max_stretch, current)
            else:
                if current >= 5:
                    lengths.append(current)
                current = 0
        if current >= 5:
            lengths.append(current)
        max_stretch = max(max_stretch, current)   # in case the last run is <5

        feats['Max_Hydrophobic_Stretch'] = max_stretch
        feats['Mean_Hydrophobic_Stretch_Length'] = np.mean(lengths) if lengths else 0.0

        feats['Mean_WALTZ'] = np.mean([self.biophys.amyloid_propensity.get(aa, 0.5) for aa in seq])
        return feats

    def _compute_scd(self, seq: str) -> float:
        L = len(seq)
        q = np.zeros(L, dtype=np.float32)
        for i, aa in enumerate(seq):
            if aa == 'K' or aa == 'R':
                q[i] = 1.0
            elif aa == 'D' or aa == 'E':
                q[i] = -1.0
        i_idx, j_idx = np.triu_indices(L, k=1)
        seps = (j_idx - i_idx).astype(np.float32)
        with np.errstate(divide='ignore'):
            contrib = q[i_idx] * q[j_idx] / np.sqrt(seps)
        return float(np.sum(contrib)) / L          # divide by L

    def _charge_asymmetry(self, seq: str) -> float:
        L = len(seq)
        mid = L // 2
        n_half = seq[:mid]
        c_half = seq[mid:]
        n_chg = sum(1 for aa in n_half if aa in 'KR') - sum(1 for aa in n_half if aa in 'DE')
        c_chg = sum(1 for aa in c_half if aa in 'KR') - sum(1 for aa in c_half if aa in 'DE')
        return (n_chg - c_chg) / L if L > 0 else 0.0

    def _aggregation_score(self, per_res: np.ndarray) -> Dict[str, float]:
        idx_hydro = self.per_residue_names.index('Local_Hydrophobicity')
        idx_sheet = self.per_residue_names.index('Local_Sheet_Propensity')
        idx_charge = self.per_residue_names.index('Local_Charge')
        hydro = per_res[:, idx_hydro]
        sheet = per_res[:, idx_sheet]
        chg   = per_res[:, idx_charge]
        agg = hydro + sheet - 0.5 * np.abs(chg)

        feats = {}
        feats['Agg_Max'] = np.max(agg)
        feats['Agg_Top5_Mean'] = np.mean(np.sort(agg)[-5:]) if len(agg) >= 5 else np.mean(agg)
        threshold = 0.5
        hotspots = np.where(agg > threshold)[0]
        feats['Agg_Hotspot_Count'] = len(hotspots)
        if len(hotspots) > 1:
            feats['Agg_Hotspot_Variance'] = np.var(hotspots)
        else:
            feats['Agg_Hotspot_Variance'] = 0.0
        feats['Agg_Mean'] = np.mean(agg)
        return feats

    def encode_protein(self, seq_id: str, sequence: str) -> np.ndarray:
        seq = sequence.upper()
        L = len(seq)

        per_res = self.idp_extractor.extract_features(seq)  # (L, 60)
        if per_res.shape[1] != self.n_per_residue:
            raise ValueError(
                f"extract_features() returned {per_res.shape[1]} columns but "
                f"per_residue_names defines {self.n_per_residue}; column/name order is out of sync"
            )
        d: Dict[str, float] = {}

        # --- per-residue statistics (only the retained mean/std/min/max) ---
        for name, stats in self._selected_stats.items():
            col = per_res[:, self._pr_idx[name]]
            for st in stats:
                d[f"{name}_{st}"] = self._STAT_FN[st](col)

        # --- surface-exposed residue means ---
        surface_mask = per_res[:, self._pr_idx['Surface_Proxy']] > 0.5
        any_surface = bool(np.any(surface_mask))
        for fname in self._selected_surf:
            vals = per_res[:, self._pr_idx[fname]]
            d[f"{fname}_surf"] = np.mean(vals[surface_mask]) if any_surface else 0.0

        # --- N/C-terminal window means ---
        n_term_idx = list(range(min(30, L)))
        c_term_idx = list(range(max(0, L - 30), L))
        for fname in self._selected_nterm:
            d[f"{fname}_Nterm"] = np.mean(per_res[n_term_idx, self._pr_idx[fname]])
        for fname in self._selected_cterm:
            d[f"{fname}_Cterm"] = np.mean(per_res[c_term_idx, self._pr_idx[fname]])

        # --- global sequence features (10 retained) ---
        hyd = np.array([self.biophys.hydrophobicity.get(aa, 0.0) for aa in seq])
        chg = np.array([self.biophys.charge.get(aa, 0.0) for aa in seq])
        d.update(self._compute_global_features(seq, hyd, chg))

        # --- charge-patterning scalars ---
        d['SCD'] = self._compute_scd(seq)
        d['Charge_Asymmetry'] = self._charge_asymmetry(seq)

        # --- aggregation-hotspot features (all 5 retained) ---
        d.update(self._aggregation_score(per_res))

        # --- explicit terminal charge / hydrophobicity ---
        if 'Nterm_Charge' in self._wanted:
            d['Nterm_Charge'] = np.sum(chg[:min(10, L)]) if L >= 1 else 0.0
        if 'Cterm_Charge' in self._wanted:
            d['Cterm_Charge'] = np.sum(chg[max(0, L - 10):]) if L >= 1 else 0.0
        if 'Nterm_HydroFraction' in self._wanted:
            d['Nterm_HydroFraction'] = np.mean(
                [1.0 if aa in 'AVILMFYW' else 0.0 for aa in seq[:min(20, L)]]) if L >= 1 else 0.0
        if 'Cterm_HydroFraction' in self._wanted:
            d['Cterm_HydroFraction'] = np.mean(
                [1.0 if aa in 'AVILMFYW' else 0.0 for aa in seq[max(0, L - 20):]]) if L >= 1 else 0.0

        return np.array([d[n] for n in SELECTED_FEATURE_NAMES], dtype=np.float32)


class SolubilityEncoderBatch:
    """Thin per-process wrapper so `Parallel(...)` workers each build one encoder."""

    def __init__(self):
        self.encoder = SolubilityEncoder()

    def encode_protein(self, seq_id, sequence):
        return self.encoder.encode_protein(seq_id, sequence)


def run_solubility_pipeline(input_csv: str, output_csv: str,
                             chunk_size: int = 100, n_jobs: int = 4):
    """Encode every protein in `input_csv` into the 200 BioGraphX-Sol
    features and write `output_csv`.

    `input_csv` must have `gene` (used as the output `ID`) and `sequence`
    columns; an optional `solubility` column is carried through as `Label`.
    """
    df = pd.read_csv(input_csv)
    seqs = df['sequence'].tolist()
    ids = df['gene'].tolist()
    labels = df['solubility'].tolist() if 'solubility' in df.columns else None

    indices = list(range(len(ids)))
    batches = [indices[i:i+chunk_size] for i in range(0, len(indices), chunk_size)]

    encoder_batch = SolubilityEncoderBatch()

    def process_batch(batch_indices):
        results = []
        for i in batch_indices:
            vec = encoder_batch.encode_protein(ids[i], seqs[i])
            results.append(vec)
        return results

    encoded_batches = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_batch)(batch) for batch in batches
    )
    all_vectors = []
    for batch_vectors in encoded_batches:
        all_vectors.extend(batch_vectors)

    data = {'ID': ids}
    if labels is not None:
        data['Label'] = labels
    for i, name in enumerate(SOLUBILITY_FEATURE_NAMES):
        data[name] = [vec[i] for vec in all_vectors]
    out_df = pd.DataFrame(data)
    out_df.to_csv(output_csv, index=False)
    print(f"Solubility encoding complete. {len(all_vectors)} proteins -> {output_csv}")
