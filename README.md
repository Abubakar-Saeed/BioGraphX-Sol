# BioGraphX-Sol

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**BioGraphX-Sol** is a physicochemical constraint-graph encoder for protein solubility prediction. It converts a protein sequence into 200 per-protein biophysical features (graph topology, frustration, polymer physics, aggregation-hotspot and charge-patterning descriptors aggregated over the sequence) and combines this biophysics stream with ESM-2 evolutionary embeddings, either through a learned per-protein gate or a post-hoc weighted blend.

By keeping the two information sources separate until they are combined, BioGraphX-Sol lets you measure - rather than assume - how much solubility prediction relies on physics versus evolutionary signal. Evaluated against the ProtSATT benchmark protocol (Deng et al., 2026) on eSOL solubility data, with zero-shot transfer to *S. cerevisiae*.

**Primary model:** the **XGBoost + ESM-only NN post-hoc ensemble** (`training/alpha_sweep_xgb_esm.py`), which essentially matches ProtSATT on eSOL (R² = 0.5451 vs. 0.5450) with a fraction of the trainable parameters (~1.44M) and no structure prediction. Physicochemical features alone (plain XGBoost) already exceed FGNNSol, a method that relies on AlphaFold3-predicted structure.

## What's in this repository

| Component | Description |
|---|---|
| `BioGraphX-Sol-Encoding/` | The 200-feature biophysical encoder (graph engine, frustration, aggregation hotspots, sticker/spacer patterning, MoRF propensity, charge decoration). |
| `esm_embeddings.py` | Per-residue ESM-2 embedding generation for a CSV of sequences (identical across the BioGraphX-* repos). |
| `training/` | Training code for the 5-seed XGBoost baseline, the jointly-trained graph_only / esm_only / gated_hybrid neural configurations, and the post-hoc XGB+ESM ensemble (primary configuration). |
| `inference.py` | Unified prediction script - pick a trained checkpoint (or the ensemble) and get a solubility probability for new proteins. |
| `Encoded Data/` | The eSOL train/test and *S. cerevisiae* CSVs, already run through the encoder (200 features + `Label`). |
| `Models Weights/` | 5-seed trained checkpoints for every configuration, ready to use with `inference.py` without retraining. |


## Repository structure

```
BioGraphX-Sol/
├── BioGraphX-Sol-Encoding/
│   ├── src/biographx_sol/
│   │   ├── __init__.py
│   │   ├── biophysics.py                 # BioPhysicsStrategy - interaction rules & AA property scales
│   │   ├── graph_engine.py               # GraphEngine - residue interaction graph + topology features
│   │   ├── frustration_analyzer.py       # FrustrationAnalyzer - configurational frustration per residue
│   │   ├── conservation_proxy.py         # ConservationProxy - alignment-free local-entropy proxy
│   │   ├── interface_profiler.py         # InteractionMotifScanner, InterfacePropensityEstimator
│   │   ├── residue_feature_extractor.py  # IDPFeatureExtractor - 60-column per-residue feature stack
│   │   ├── solubility_encoder.py         # SolubilityEncoder(Batch), run_solubility_pipeline - aggregates
│   │   │                                 # the per-residue stack into the 200 per-protein features
│   │   └── utils/feature_names.py        # SOLUBILITY_FEATURE_NAMES - the 200 output columns, in order
│   └── run.py                       # CLI: sequences (CSV) -> 200-feature CSV
├── esm_embeddings.py                 # CLI: sequences (CSV) -> per-protein ESM-2 .npz embeddings
├── training/
│   ├── train_xgboost_esol.py         # 200 features -> 5-seed XGBoost
│   ├── train_nn_esol.py              # 200 features + ESM-2 -> graph_only / esm_only / gated_hybrid
│   ├── alpha_sweep_xgb_esm.py        # post-hoc XGBoost + ESM-only ensemble (loads saved checkpoints, no retraining)
│   └── alpha_sweep_graph_esm.py      # post-hoc graph_only + ESM-only ensemble (same, held-out-slice alpha selection)
├── inference.py                      # Unified inference + zero-shot evaluation across all 5 configurations
├── Encoded Data/
│   ├── eSol_train_encoded.csv        # 2019 proteins
│   ├── eSol_test_encoded.csv         # 660 proteins
│   └── S.cerevisiae_test_encoded.csv # 108 proteins (zero-shot external test)
├── Models Weights/
│   ├── xgboost_models/               # seed_{2024..2028}.json
│   ├── graph_only/                   # model_seed{s}_fold{k}.pt, model_seed{s}_refit.pt, summary.csv
│   ├── esm_only/                     # (same layout)
│   ├── gated_hybrid/                 # (same layout) + gate_history.csv
│   └── ensemble/                     # alpha_sweep_seed{s}.csv, metrics_per_seed.csv, summary.csv
├── requirements.txt
├── LICENSE
└── README.md
```

**Why `train_nn_esol.py` trains graph_only, esm_only and gated_hybrid together, not one script per model (BioGraphX-IDP convention):** they share one 5-fold cross-validation split, so training them in one pass avoids reloading each other's predictions from disk.

**Ensembling:** BioGraphX-Sol's ensemble blends a physics-stream prediction and an ESM-stream prediction with a weight alpha (`alpha * physics + (1 - alpha) * ESM`). Alpha is selected **per seed**, by grid search over `{0.00, 0.05, ..., 1.00}` on a held-out validation slice (`--val-size`, default the first 268 of the 660 test rows); each seed's own alpha forms that seed's ensemble prediction, and reported metrics are the mean +/- SD across the 5 per-seed ensembles. `alpha_sweep_xgb_esm.py` applies this to XGBoost + ESM-only (the primary configuration); `alpha_sweep_graph_esm.py` applies the same procedure to graph_only + ESM-only. Both need no retraining - just the already-saved checkpoints.

## Requirements

- Python 3.9+
- A CUDA GPU is strongly recommended for ESM-2 embedding generation and for training the neural models (the ESM-2 3B checkpoint used here is CPU-feasible only for a handful of sequences).

```bash
pip install -r requirements.txt
```

`python-igraph` requires a C toolchain on some platforms; if the wheel install fails, see the [python-igraph install docs](https://python.igraph.org/en/stable/install.html).

## 1. Feature encoding

Input CSV format:

```
gene,sequence,solubility
P0A6F5,MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAP...,0.82
```

`solubility` is optional - omit it to encode unlabeled sequences for inference. `gene` becomes the output `ID`.

```bash
cd BioGraphX-Sol-Encoding
python run.py --input-file eSol_train.csv \
              --output-file eSol_train_encoded.csv \
              --n-jobs 8
```

Output: one row per protein, `ID, [Label,] <200 feature columns>`. The 200 feature names are exported as `biographx_sol.SOLUBILITY_FEATURE_NAMES`.

### The 200 features

| Group | Description |
|---|---|
| Graph topology & frustration | Degree, weighted degree, betweenness, closeness, clustering and hybrid-interaction ratio from the residue-interaction graph, plus configurational frustration - summarized per protein (mean/std/min/max, surface-exposed subset, N-/C-terminal windows). |
| Sequence & physicochemical | Local hydrophobicity/charge/helix/sheet propensity, disorder-promoting-residue scale, secondary-structure prediction fractions, bulkiness, Ramachandran flexibility, autocorrelations - same per-protein summaries. |
| Polymer physics & charge patterning | FCR, NCPR, polyampholyte score, charge-patterning kappa, sequence charge decoration (SCD), N/C-terminal charge asymmetry, sticker/spacer density and dispersion, multi-scale sequence complexity (windows 5/10/25/50) and its cross-scale variance. |
| Aggregation & interface | MoRF core/flank/dip score, interface propensity, motif density, and 5 aggregation-hotspot features (a hydrophobicity + sheet-propensity - |charge| score used to flag aggregation-prone stretches: max, top-5 mean, hotspot count and position variance, overall mean). |
| Global sequence properties | Isoelectric point, net charge, charge density, aromaticity, instability index, aliphatic index, cysteine count, hydrophobic-stretch statistics, mean WALTZ/amyloid propensity. |
| **Total** | **200** - a statistical-summary representation over the same per-residue signal family BioGraphX-IDP uses for disorder prediction, computed entirely from sequence with no dependence on experimental or predicted 3D structure. |

You can also use the encoder directly from Python:

```python
import sys; sys.path.insert(0, "BioGraphX-Sol-Encoding/src")
from biographx_sol import SolubilityEncoder

encoder = SolubilityEncoder()
features = encoder.encode_protein("P0A6F5", "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAP")  # (200,) float32 array
```

## 2. ESM-2 embedding generation

```bash
python esm_embeddings.py --csv-path eSol_train_for_esm.csv \
                          --output-dir esm_embeddings/eSol_train \
                          --model-name facebook/esm2_t36_3B_UR50D \
                          --batch-size 8
```

Writes one compressed `{ID}.npz` file per protein (key `embedding`, shape `[seq_len, hidden_dim]`, `float16`), mean-pooled over residues at load time. This script needs `ID` and `Sequence` columns - rename BioGraphX-Sol's own `gene`/`sequence` columns first so the `.npz` filenames line up with the `ID` written by the encoder. Large datasets can be split across multiple runs/GPUs with `--part` / `--total-parts`.

## 3. Model training

```bash
# 5-seed XGBoost baseline
python training/train_xgboost_esol.py \
    --train-csv "Encoded Data/eSol_train_encoded.csv" \
    --test-csv "Encoded Data/eSol_test_encoded.csv" \
    --model-dir models_protsatt \
    --importance-dir importance_protsatt

# graph_only / esm_only / gated_hybrid, jointly (shared 5-fold CV)
python training/train_nn_esol.py \
    --train-csv "Encoded Data/eSol_train_encoded.csv" \
    --test-csv "Encoded Data/eSol_test_encoded.csv" \
    --esm-dirs esm_embeddings/eSol_train esm_embeddings/eSol_test \
    --output-dir biographx_sol_nn_results
```

Both scripts run 5 seeds (`--seeds`, default `2024 2025 2026 2027 2028`) with the ProtSATT protocol: 5-fold CV inside the 2019-protein train split to pick a stopping point (best `n_estimators` / epoch), then a refit on the full train set, evaluated once on the full 660-protein test set. `train_nn_esol.py` also writes the fitted `StandardScaler` (`scaler.pkl`) - `inference.py` needs this to normalize new proteins the same way the physics features were normalized at training time.

Pretrained checkpoints for all 5 configurations are already included in `Models Weights/` - you don't need to retrain to use `inference.py`.

### Post-hoc ensembles (no retraining)

Re-blend already-trained checkpoints - useful for re-weighting an ensemble without rerunning 5-fold CV:

```bash
# XGBoost + ESM-only
python training/alpha_sweep_xgb_esm.py \
    --test-csv "Encoded Data/eSol_test_encoded.csv" \
    --xgb-model-dir "Models Weights/xgboost_models" \
    --esm-model-dir "Models Weights/esm_only" \
    --esm-dirs esm_embeddings/eSol_test \
    --output-dir xgb_esm_ensemble

# graph_only + ESM-only
python training/alpha_sweep_graph_esm.py \
    --test-csv "Encoded Data/eSol_test_encoded.csv" \
    --graph-model-dir "Models Weights/graph_only" \
    --esm-model-dir "Models Weights/esm_only" \
    --scaler-path biographx_sol_nn_results/scaler.pkl \
    --esm-dirs esm_embeddings/eSol_test \
    --output-dir graph_esm_ensemble
```

Each writes `alpha_sweep.csv` (mean +/- SD across seeds, for every alpha tried) and `metrics_per_seed.csv` / `summary.csv` (the per-seed ensemble at its own selected alpha, and the mean +/- SD across seeds).

## 4. Inference and zero-shot evaluation

`inference.py` consumes a **pre-encoded** CSV (produced by `BioGraphX-Sol-Encoding/run.py`) - unlike BioGraphX-IDP's inference script, it doesn't encode raw sequences on the fly, matching how these models were actually trained and evaluated.

```bash
# Gated Hybrid (biophysics + ESM-2, learned gate)
python inference.py --model-type gated_hybrid \
    --input-csv new_proteins_encoded.csv \
    --model-dir "Models Weights" \
    --esm-dir esm_embeddings/new \
    --output-csv predictions.csv

# XGBoost baseline (no ESM-2 required)
python inference.py --model-type xgboost \
    --input-csv new_proteins_encoded.csv \
    --model-dir "Models Weights" \
    --output-csv predictions.csv

# graph_only / esm_only / ensemble work the same way (--model-type ...)
```

Every configuration averages predictions over the 5 seed checkpoints. `--model-type ensemble` blends graph_only and esm_only using the mean of the per-seed alphas recorded in `Models Weights/ensemble/metrics_per_seed.csv` (override with `--alpha`).

Pass `--eval` when the input CSV has a `Label`/`solubility` column to also compute and save R2, Pearson, RMSE, ACC, Precision, Recall, F1, AUC and MCC - this runs the zero-shot *S. cerevisiae* evaluation directly:

```bash
python inference.py --model-type ensemble \
    --input-csv "Encoded Data/S.cerevisiae_test_encoded.csv" \
    --model-dir "Models Weights" \
    --esm-dir esm_embeddings/S.cerevisiae_test \
    --output-csv sc_zeroshot_predictions.csv --eval
```

## Notes

- The 200-feature encoder and the ESM-2 mean-pooling both operate purely on sequence - no experimental or predicted structure is used anywhere in the pipeline.
- The eSOL 2019/660 split and the 108-protein *S. cerevisiae* zero-shot benchmark follow the GATSol/ProtSATT protocol exactly, so results are directly comparable to published baselines on the same splits.
- `BioGraphX-Sol-Encoding`'s output header was verified column-for-column against the committed `Encoded Data/*.csv` files; end-to-end plumbing (encode -> train -> save checkpoint -> load checkpoint -> infer/ensemble, for all configurations) was verified by running the actual scripts, not just import checks. Full numeric reproduction against the exact reference values could not be verified in this session because the raw (pre-encoding) sequence CSVs were not available locally - only the already-encoded feature CSVs were.

## License

Released under the [MIT License](LICENSE).

## Related work

- [BioGraphX](https://github.com/Abubakar-Saeed/BioGraphX) - the physicochemical graph encoding framework this project builds on, applied to protein subcellular localization prediction.
- [BioGraphX-RNA](https://github.com/Abubakar-Saeed/BioGraphX-RNA) - the same graph-encoding approach applied to RNA sequences for RNA subcellular localization prediction.
- [BioGraphX-IDP](https://github.com/Abubakar-Saeed/BioGraphX-IDP) - the same graph-encoding approach applied to intrinsic disorder prediction, with a per-residue (rather than per-protein) feature representation.
