"""
SOLUBILITY_FEATURE_NAMES
================================================================================

Author: Abubakar Saeed

Description:
    The 200 per-protein features actually kept for the BioGraphX-Sol
    solubility models, in the exact column order written by
    SolubilityEncoder.encode_protein(). Everything else the underlying
    per-residue extractor could statistically summarize is no longer
    computed - this list is also used to derive, at encoder-construction
    time, exactly which of the 60 per-residue signals need which
    mean/std/min/max/surface/N-term/C-term summaries.
"""

SELECTED_FEATURE_NAMES = [
    'Contact_Order_surf', 'Agg_Hotspot_Count', 'Betweenness_std', 'Local_Kappa_min',
    'Betweenness_Log_max', 'pI', 'Disorder_AA_Scale_Local_mean', 'Sticker_Density_surf',
    'Contact_Order_Cterm', 'Agg_Hotspot_Variance', 'Disorder_AA_Scale_surf', 'Contact_Order_mean',
    'Mean_WALTZ', 'Sticker_Density_max', 'Betweenness_max', 'Polyampholyte_Score_min',
    'Sticker_Density_mean', 'Charge_Density', 'Polyampholyte_Score_mean', 'FCR_mean',
    'FCR_min', 'Local_Charge_LargeWin_mean', 'Local_Sheet_Propensity_surf', 'Sticker_Dispersion_mean',
    'Disorder_AA_Scale_mean', 'NCPR_mean', 'Predicted_Coil_max', 'Local_Charge_surf',
    'Local_Sheet_Propensity_mean', 'Global_Disorder_Composition_Bias_max', 'Contact_Order_Nterm', 'OrderDisorder_Pattern_Entropy_mean',
    'Net_Charge_pH7', 'Sticker_Density_std', 'Net_Charge_Global_max', 'Local_Sheet_Propensity_std',
    'Predicted_Sheet_mean', 'Weighted_Degree_max', 'Degree_max', 'Local_Charge_mean',
    'Closeness_Log_mean', 'CTerm_Distance_mean', 'Weighted_Degree_Log_max', 'Uversky_Distance_mean',
    'Predicted_Helix_mean', 'Disorder_AA_Scale_Nterm', 'Closeness_std', 'Aromaticity',
    'Charge_Autocorr_Lag1_std', 'Agg_Top5_Mean', 'Net_Charge_Global_min', 'Global_Disorder_Composition_Bias_min',
    'CTerm_Distance_std', 'Closeness_Log_max', 'Net_Charge_Global_mean', 'Local_Kappa_mean',
    'Complexity_W50_mean', 'OrderDisorder_Pattern_Entropy_std', 'Local_Sheet_Propensity_min', 'Local_Edge_Density_mean',
    'Complexity_W50_min', 'Spacer_Density_surf', 'Local_Edge_Density_std', 'Global_Disorder_Composition_Bias_mean',
    'Betweenness_Log_std', 'Local_Hydrophobicity_Nterm', 'Closeness_min', 'In_Motif_std',
    'Uversky_Distance_std', 'Degree_Log_max', 'Complexity_W5_std', 'Spacer_Density_mean',
    'Closeness_max', 'Agg_Mean', 'Spacer_Density_max', 'Agg_Max',
    'Sticker_Dispersion_min', 'SCD', 'Charge_Autocorr_Lag1_max', 'Weighted_Degree_Log_mean',
    'Disorder_AA_Scale_Local_std', 'OrderDisorder_Pattern_Entropy_min', 'Weighted_Degree_Log_std', 'Interface_Propensity_std',
    'Sticker_Dispersion_std', 'NTerm_Distance_mean', 'Sticker_Density_min', 'MoRF_Flank_Score_min',
    'Order_Res_Fraction_max', 'Instability_Index', 'Complexity_W25_min', 'NTerm_Distance_std',
    'Disorder_AA_Scale_Local_min', 'Complexity_Std_Win7_min', 'Degree_std', 'Local_Sheet_Propensity_max',
    'NCPR_std', 'Local_Sheet_Propensity_Nterm', 'MoRF_Flank_Score_std', 'Nterm_HydroFraction',
    'Interface_Propensity_max', 'Disorder_Res_Fraction_min', 'Contact_Order_std', 'Local_Edge_Density_max',
    'Local_Hydrophobicity_std', 'Length_Normalized_std', 'Complexity_W5_min', 'Local_Charge_Cterm',
    'Disorder_AA_Scale_std', 'Local_Hydrophobicity_Cterm', 'Uversky_Distance_max', 'Cysteine_Count',
    'Complexity_W50_max', 'Global_Disorder_Composition_Bias_std', 'Disorder_from_Ifprop_std', 'ProGly_Density_max',
    'Local_Bulkiness_min', 'Local_Charge_min', 'Frustration_Cterm', 'Charge_Asymmetry',
    'Complexity_Scale_Variance_min', 'Order_Res_Fraction_mean', 'Nterm_Charge', 'Local_Charge_std',
    'Aliphatic_Index', 'Degree_Log_mean', 'ProGly_Density_mean', 'Local_Kappa_max',
    'Complexity_W5_mean', 'Frustration_std', 'Local_Hydrophobicity_max', 'Local_Helix_Propensity_max',
    'MoRF_Dip_Score_max', 'Local_Edge_Density_min', 'Frustration_Nterm', 'Hybrid_Ratio_Nterm',
    'Surface_Proxy_std', 'Interface_Propensity_min', 'Closeness_Log_min', 'Complexity_Scale_Variance_std',
    'Frustration_surf', 'MoRF_Dip_Score_min', 'Local_Charge_LargeWin_max', 'Local_Helix_Propensity_min',
    'Local_Hydrophobicity_min', 'Local_Sheet_Propensity_Cterm', 'MoRF_Core_Score_std', 'Weighted_Degree_min',
    'Mean_Hydrophobic_Stretch_Length', 'Local_Motif_Density_mean', 'Local_Bulkiness_mean', 'Local_Hydrophobicity_surf',
    'Predicted_Coil_std', 'Hydro_Autocorr_Lag1_max', 'Local_Bulkiness_max', 'Complexity_W25_std',
    'Predicted_Coil_mean', 'Sticker_Density_Cterm', 'Ramachandran_Flexibility_std', 'ProGly_Density_std',
    'Hydro_Autocorr_Lag1_std', 'Local_Motif_Density_std', 'Charge_Autocorr_Lag1_mean', 'Spacer_Density_std',
    'Sticker_Density_Nterm', 'Clustering_std', 'Spacer_Density_Nterm', 'Disorder_from_Ifprop_max',
    'Polyampholyte_Score_std', 'Local_Charge_Nterm', 'Cterm_Charge', 'Predicted_Helix_std',
    'Complexity_Std_Win7_max', 'Order_Res_Fraction_std', 'Uversky_Distance_min', 'Surface_Proxy_mean',
    'Weighted_Degree_Log_min', 'Complexity_W50_std', 'Disorder_AA_Scale_Local_max', 'Local_Charge_max',
    'MoRF_Dip_Score_mean', 'GRAVY_Global_min', 'MoRF_Core_Score_Cterm', 'Local_Helix_Propensity_std',
    'Net_Charge_Global_std', 'Max_Hydrophobic_Stretch', 'MoRF_Flank_Score_max', 'Betweenness_Log_mean',
    'Frustration_Entropy_Win7_std', 'MoRF_Flank_Score_mean', 'Sequence_Complexity_std', 'Disorder_from_Ifprop_mean',
    'MoRF_Core_Score_Nterm', 'Ramachandran_Flexibility_mean', 'Weighted_Degree_std', 'Surface_Proxy_max',
    'Closeness_Log_std', 'Interface_Propensity_mean', 'Sequence_Complexity_max', 'Disorder_AA_Scale_Cterm',
]
assert len(SELECTED_FEATURE_NAMES) == 200, len(SELECTED_FEATURE_NAMES)

# Alias used by the rest of the package / downstream consumers.
SOLUBILITY_FEATURE_NAMES = list(SELECTED_FEATURE_NAMES)
