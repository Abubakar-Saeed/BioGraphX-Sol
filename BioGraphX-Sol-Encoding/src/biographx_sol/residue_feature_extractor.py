"""
IDPFeatureExtractor
================================================================================

Author: Abubakar Saeed

Description:
    Per-residue feature extractor (60 columns): wires BioPhysicsStrategy,
    GraphEngine, FrustrationAnalyzer, ConservationProxy,
    InteractionMotifScanner and InterfacePropensityEstimator together into
    graph topology, frustration, disorder/order composition, polymer-physics
    (MoRF propensity, FCR/NCPR/polyampholyte, charge patterning kappa,
    sticker/spacer density, contact order) and local biophysical-scale
    features. This is an intermediate, per-residue representation - it is
    consumed by SolubilityEncoder, which aggregates it into the 200
    per-protein features actually used by the models.

    Named `N_PER_RESIDUE` = 60; must stay equal to the number of arrays
    stacked in `extract_features()` and to `SolubilityEncoder.per_residue_names`.
"""

import numpy as np

from .biophysics import BioPhysicsStrategy
from .graph_engine import GraphEngine
from .frustration_analyzer import FrustrationAnalyzer
from .conservation_proxy import ConservationProxy
from .interface_profiler import InteractionMotifScanner, InterfacePropensityEstimator

# Number of arrays stacked by IDPFeatureExtractor.extract_features(); must stay
# equal to len(SolubilityEncoder.per_residue_names). Bump deliberately on change.
N_PER_RESIDUE = 60


class IDPFeatureExtractor:
    def __init__(self):
        self.biophys = BioPhysicsStrategy()
        self.graph_engine = GraphEngine(self.biophys)
        self.frustration_analyzer = FrustrationAnalyzer(self.biophys)
        self.cons_proxy = ConservationProxy()
        self.motif_scanner = InteractionMotifScanner()
        self.ifprop_est = InterfacePropensityEstimator(
            self.biophys, self.cons_proxy, self.motif_scanner)

    def extract_features(self, sequence: str) -> np.ndarray:
        L = len(sequence)
        seq = sequence.upper()

        def safe_convolve(arr, kernel):
            if len(arr) >= len(kernel):
                return np.convolve(arr, kernel, mode='same')
            else:
                return arr

        g = self.graph_engine.build_complete_graph(seq)
        gf = self.graph_engine.extract_per_residue_graph_features(g)
        degree = gf["degree"]
        w_degree = gf["weighted_degree"]
        between = gf["betweenness"]
        close = gf["closeness"]
        clustering = gf["clustering"]
        hybrid_ratio = gf["hybrid_ratio"]

        frustration = self.frustration_analyzer.compute_frustration(g)
        complexity = self.cons_proxy.per_residue_entropy(seq)

        ifprop, surface_proxy = self.ifprop_est.compute_per_residue(g, seq)
        disorder_from_ifprop = 1.0 - ifprop

        motif_flags = self.motif_scanner.get_per_residue_motif_flags(seq).astype(float)

        kernel7 = np.ones(7) / 7
        kernel15 = np.ones(15) / 15
        kernel51 = np.ones(51) / 51

        local_motif_density = safe_convolve(motif_flags, kernel7)
        disorder_aa = np.array([self.biophys.disorder_aa_scale.get(aa, 0.0) for aa in seq], dtype=np.float32)
        disorder_aa_local = safe_convolve(disorder_aa, kernel7)

        hyd = np.array([self.biophys.hydrophobicity.get(aa, 0.0) for aa in seq])
        chg = np.array([self.biophys.charge.get(aa, 0.0) for aa in seq])
        hyd_local = safe_convolve(hyd, kernel7)
        chg_local = safe_convolve(chg, kernel7)
        chg_large = safe_convolve(chg, kernel15)

        helix_raw = np.array([self.biophys.helix_propensity.get(aa, 0.0) for aa in seq])
        sheet_raw = np.array([self.biophys.sheet_propensity.get(aa, 0.0) for aa in seq])
        helix_local = safe_convolve(helix_raw, kernel7)
        sheet_local = safe_convolve(sheet_raw, kernel7)

        pred_helix = np.zeros(L, dtype=np.float32)
        pred_sheet = np.zeros(L, dtype=np.float32)
        pred_coil  = np.zeros(L, dtype=np.float32)
        for i in range(L):
            h = helix_local[i]
            s = sheet_local[i]
            if h > s:
                pred_helix[i] = 1.0
            elif s > h:
                pred_sheet[i] = 1.0
            else:
                pred_coil[i] = 1.0

        bulk = np.array([self.biophys.bulkiness.get(aa, 150.0) for aa in seq])
        bulk_local = safe_convolve(bulk, kernel7)

        disorder_res = set('AGPSQEKR')
        order_res = set('ILVFWCYM')
        dis_frac = np.zeros(L, dtype=np.float32)
        ord_frac = np.zeros(L, dtype=np.float32)
        for i in range(L):
            start = max(0, i-3)
            end = min(L, i+4)
            win_size = end - start
            dis_frac[i] = sum(1 for r in seq[start:end] if r in disorder_res) / win_size
            ord_frac[i] = sum(1 for r in seq[start:end] if r in order_res) / win_size

        n_term_dist = np.array([i/L for i in range(L)], dtype=np.float32)
        c_term_dist = np.array([(L-1-i)/L for i in range(L)], dtype=np.float32)

        gravy = np.mean(hyd)
        net_charge_global = np.sum(chg)
        length_norm = L / 500.0
        total_dis = sum(1 for r in seq if r in disorder_res)
        total_ord = sum(1 for r in seq if r in order_res)
        global_disorder_ratio = total_dis / (total_ord + 1e-10)

        def local_autocorr(arr, half_win=3):
            ac = np.zeros_like(arr)
            for i in range(L):
                start = max(0, i-half_win)
                end = min(L, i+half_win+1)
                win = arr[start:end]
                if len(win) < 3:
                    ac[i] = 0.0
                else:
                    m = np.mean(win)
                    v = np.var(win)
                    if v < 1e-6:
                        ac[i] = 0.0
                    else:
                        c = np.sum((win[:-1]-m)*(win[1:]-m))
                        ac[i] = c / ((len(win)-1)*v)
            return ac
        hydro_autocorr = local_autocorr(hyd)
        charge_autocorr = local_autocorr(chg)

        frust_entropy = np.zeros(L, dtype=np.float32)
        for i in range(L):
            start = max(0, i-3)
            end = min(L, i+4)
            win = frustration[start:end]
            if len(win) < 2:
                frust_entropy[i] = 0.0
            else:
                hist, _ = np.histogram(win, bins=5, range=(0,1))
                if hist.sum() > 0:
                    hist = hist / hist.sum()
                    frust_entropy[i] = -np.sum(hist * np.log(hist + 1e-10))
                else:
                    frust_entropy[i] = 0.0

        abs_chg_local = np.abs(chg_local)
        uversky_dist = abs_chg_local - (2.785 * hyd_local - 1.151)

        complexity_std = np.zeros(L, dtype=np.float32)
        for i in range(L):
            start = max(0, i-3)
            end = min(L, i+4)
            win = complexity[start:end]
            complexity_std[i] = np.std(win) if len(win)>1 else 0.0

        # --- B1 fix: local edge density using neighbor sets (no dense adj) ---
        neighbors = {v: set(g.neighbors(v)) for v in range(L)}
        local_edge_density = np.zeros(L, dtype=np.float32)
        win_radius = 10
        for i in range(L):
            start = max(0, i - win_radius)
            end = min(L, i + win_radius + 1)
            indices = list(range(start, end))
            k = len(indices)
            if k < 2:
                continue
            n_edges = 0
            for a_idx, a in enumerate(indices):
                for b in indices[a_idx+1:]:
                    if b in neighbors[a]:
                        n_edges += 1
            max_edges = k * (k - 1) // 2
            local_edge_density[i] = n_edges / max_edges if max_edges > 0 else 0.0

        od_pattern_entropy = np.zeros(L, dtype=np.float32)
        for i in range(L):
            start = max(0, i-5)
            end = min(L, i+6)
            win = disorder_aa[start:end]
            binary = (win > 0.3).astype(int)
            if len(binary) < 2:
                continue
            cnt = np.bincount(binary, minlength=2)
            p = cnt / cnt.sum()
            od_pattern_entropy[i] = -np.sum(p * np.log(p + 1e-10))

        kappa = np.zeros(L, dtype=np.float32)
        global_charge_asym = np.abs(np.sum(chg)) / L
        for i in range(L):
            start = max(0, i-7)
            end = min(L, i+8)
            win_chg = chg[start:end]
            delta = np.sum((np.abs(win_chg) - global_charge_asym)**2) / len(win_chg)
            kappa[i] = delta / 1.0
            if kappa[i] > 1.0:
                kappa[i] = 1.0

        stickers_set = set('YFWR')
        spacers_set  = set('GSP')
        sticker_density = np.zeros(L, dtype=np.float32)
        spacer_density  = np.zeros(L, dtype=np.float32)
        sticker_dispersion = np.zeros(L, dtype=np.float32)
        for i in range(L):
            start = max(0, i-10)
            end = min(L, i+11)
            win = seq[start:end]
            n_stickers = sum(1 for r in win if r in stickers_set)
            n_spacers  = sum(1 for r in win if r in spacers_set)
            w_len = len(win)
            sticker_density[i] = n_stickers / w_len
            spacer_density[i]  = n_spacers  / w_len
            sticker_pos = [j for j, r in enumerate(win) if r in stickers_set]
            if len(sticker_pos) >= 2:
                gaps = np.diff(sticker_pos)
                sticker_dispersion[i] = np.mean(gaps) / w_len
            else:
                sticker_dispersion[i] = 1.0

        pro_gly_density = np.zeros(L, dtype=np.float32)
        for i in range(L):
            start = max(0, i-4)
            end = min(L, i+5)
            win = seq[start:end]
            pro_gly_density[i] = sum(1 for r in win if r in 'PG') / len(win)

        # MoRF
        hyd_context = safe_convolve(hyd, kernel51)
        morf_dip_score = hyd_local - hyd_context
        dis_context = safe_convolve(dis_frac, kernel51)
        morf_flank_score = dis_context
        ss_capacity = np.maximum(helix_raw, sheet_raw)
        ss_local = safe_convolve(ss_capacity, kernel7)
        ss_min, ss_max = 0.57, 1.70
        ss_local_norm = (ss_local - ss_min) / (ss_max - ss_min + 1e-10)
        morf_core_score = (
            0.40 * np.clip(morf_dip_score / 2.0, 0, 1) +
            0.35 * morf_flank_score +
            0.25 * ss_local_norm
        )
        morf_core_score = np.clip(morf_core_score, 0, 1)

        # FCR/NCPR (without H)
        pos_set = set('KR')
        neg_set = set('DE')
        fcr  = np.zeros(L, dtype=np.float32)
        ncpr = np.zeros(L, dtype=np.float32)
        polyamph = np.zeros(L, dtype=np.float32)
        half_win = 10
        for i in range(L):
            start = max(0, i - half_win)
            end   = min(L, i + half_win + 1)
            win   = seq[start:end]
            n     = len(win)
            n_pos = sum(1 for aa in win if aa in pos_set)
            n_neg = sum(1 for aa in win if aa in neg_set)
            fcr[i]  = (n_pos + n_neg) / n
            ncpr[i] = (n_pos - n_neg) / n
            polyamph[i] = fcr[i] * (1.0 - abs(ncpr[i]))

        # Multi-scale complexity
        complexity_w5  = self.cons_proxy.per_residue_entropy(seq, window_radius=5)
        complexity_w25 = self.cons_proxy.per_residue_entropy(seq, window_radius=25)
        if L >= 100:
            complexity_w50 = self.cons_proxy.per_residue_entropy(seq, window_radius=50)
        else:
            complexity_w50 = complexity_w25.copy()
        scales_matrix = np.stack([complexity_w5, complexity, complexity_w25, complexity_w50], axis=0)
        complexity_scale_variance = np.var(scales_matrix, axis=0)

        # Contact order (non-backbone)
        contact_order = np.zeros(L, dtype=np.float32)
        for v in range(L):
            inc = g.incident(v)
            long_range_seps = []
            for eid in inc:
                edge = g.es[eid]
                if edge['interaction_type'] != 'backbone':
                    other = edge.target if edge.source == v else edge.source
                    sep = abs(v - other)
                    if sep > 1:
                        long_range_seps.append(sep)
            if long_range_seps:
                contact_order[v] = np.mean(long_range_seps) / L

        # Ramachandran flexibility
        flex_raw = np.array([self.biophys.ramachandran_flex.get(aa, 0.5) for aa in seq])
        flex_local = safe_convolve(flex_raw, kernel7)

        features = np.column_stack([
            degree, w_degree, between, close, clustering, hybrid_ratio,
            frustration, complexity, disorder_aa,
            hyd_local, chg_local, helix_local, sheet_local,
            dis_frac, ord_frac,
            n_term_dist, c_term_dist,
            np.full(L, gravy), np.full(L, net_charge_global), np.full(L, length_norm),
            np.log1p(degree), np.log1p(between), np.log1p(w_degree), np.log1p(close*100),
            ifprop, disorder_from_ifprop,
            motif_flags, local_motif_density,
            hydro_autocorr, charge_autocorr,
            frust_entropy, uversky_dist, complexity_std,
            np.full(L, global_disorder_ratio),
            disorder_aa_local,
            surface_proxy,
            local_edge_density,
            pred_helix, pred_sheet, pred_coil,
            od_pattern_entropy,
            bulk_local,
            chg_large,
            kappa,
            sticker_density, spacer_density, sticker_dispersion,
            pro_gly_density,
            morf_core_score, morf_flank_score, morf_dip_score,
            fcr, ncpr, polyamph,
            complexity_w5, complexity_w25, complexity_w50, complexity_scale_variance,
            contact_order,
            flex_local
        ])
        if features.shape[1] != N_PER_RESIDUE:
            raise AssertionError(
                f"extract_features() stacked {features.shape[1]} columns, expected {N_PER_RESIDUE}"
            )
        return features.astype(np.float32)
