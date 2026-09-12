#!/usr/bin/env python3
"""
BioGraphX-Sol - unified inference / evaluation script.

Predicts solubility for new proteins using one of the five trained
BioGraphX-Sol configurations from `training/`:

    --model-type xgboost        200 biophysical features only
    --model-type graph_only     200 biophysical features only (NN)
    --model-type esm_only       ESM-2 embedding only
    --model-type gated_hybrid   200 biophysical features + ESM-2, learned per-protein gate
    --model-type ensemble       post-hoc alpha-blend of graph_only + esm_only

Unlike BioGraphX-IDP's `inference.py` (which encodes raw sequences on the
fly), BioGraphX-Sol's models were trained on features precomputed by
`BioGraphX-Sol-Encoding/run.py` - this script consumes that same encoded CSV
directly, exactly as the training/evaluation scripts in `training/` do.
Encode your sequences first:

    cd BioGraphX-Sol-Encoding
    python run.py --input-file new_proteins.csv --output-file new_proteins_encoded.csv

Every model type averages predictions over `--seeds` (5 by default, matching
the 5-seed refit checkpoints saved by training). `--model-dir` is the root
folder holding one subdirectory per configuration, i.e. the layout of the
`Models Weights/` directory in this repo:

    <model-dir>/xgboost_models/seed_<seed>.json
    <model-dir>/graph_only/model_seed<seed>_refit.pt
    <model-dir>/esm_only/model_seed<seed>_refit.pt
    <model-dir>/gated_hybrid/model_seed<seed>_refit.pt
    <model-dir>/ensemble/metrics_per_seed.csv      (per-seed alpha, for --model-type ensemble)

If the input CSV has a Label/solubility column, pass --eval to also compute
and print/save R2, Pearson, RMSE, ACC, Precision, Recall, F1, AUC and MCC -
this is what reproduces the zero-shot S. cerevisiae evaluation from the
paper (just point --input-csv at S.cerevisiae_test_encoded.csv).

Examples
--------
    # Gated Hybrid (flagship model)
    python inference.py --model-type gated_hybrid \\
        --input-csv new_proteins_encoded.csv \\
        --model-dir "Models Weights" \\
        --esm-dir esm_embeddings/new \\
        --output-csv predictions.csv

    # XGBoost baseline (no ESM-2 needed)
    python inference.py --model-type xgboost \\
        --input-csv new_proteins_encoded.csv \\
        --model-dir "Models Weights" \\
        --output-csv predictions.csv

    # Zero-shot evaluation against a labeled external test set
    python inference.py --model-type ensemble \\
        --input-csv "Encoded Data/S.cerevisiae_test_encoded.csv" \\
        --model-dir "Models Weights" \\
        --esm-dir esm_embeddings/S.cerevisiae_test \\
        --output-csv sc_zeroshot_predictions.csv --eval
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from training.train_nn_esol import GraphOnlyNN, ESMOnlyNN, GatedHybridNN, ESM_DIM  # noqa: E402

META_COLS = {"ID", "gene", "uniprot", "sequence", "pdb_filename", "Label", "solubility", "target"}
ID_CANDIDATES = ["gene", "ID", "uniprot"]
TARGET_CANDIDATES = ["Label", "solubility", "target"]
THRESHOLD = 0.5


def pick_col(cols, candidates, what, required=True):
    hit = next((c for c in candidates if c in cols), None)
    if required:
        assert hit, f"no {what} column found (looked for {candidates})"
    return hit


def load_input(input_csv):
    df = pd.read_csv(input_csv)
    id_col = pick_col(df.columns, ID_CANDIDATES, "ID")
    tgt_col = pick_col(df.columns, TARGET_CANDIDATES, "target", required=False)
    feat_cols = [c for c in df.columns
                 if c not in META_COLS and c not in (id_col, tgt_col)
                 and np.issubdtype(df[c].dtype, np.number)]
    assert len(feat_cols) == 200, f"Expected 200 features, got {len(feat_cols)}"
    return df, id_col, tgt_col, feat_cols


def build_esm_matrix(ids, esm_dir):
    mat = np.zeros((len(ids), ESM_DIM), dtype=np.float32)
    missing = 0
    for i, pid in enumerate(ids):
        path = os.path.join(esm_dir, f"{pid}.npz")
        if not os.path.exists(path):
            missing += 1
            continue
        data = np.load(path)
        key = "embedding" if "embedding" in data else data.files[0]
        emb = data[key].astype(np.float32)
        if emb.ndim == 2 and emb.shape[0] > 2:
            emb = emb[1:-1]
        mat[i] = emb.mean(axis=0) if emb.ndim == 2 else emb
    if missing:
        print(f"  [warn] {missing}/{len(ids)} ids had no ESM embedding in {esm_dir} (left as zeros)")
    return mat


def get_scaler(args, feat_cols):
    import joblib

    scaler_path = args.scaler_path or os.path.join(args.model_dir, "scaler.pkl")
    if os.path.exists(scaler_path):
        print(f"  Loaded scaler from {scaler_path}")
        return joblib.load(scaler_path)

    fallback = args.scaler_fit_csv or "eSol_train_encoded.csv"
    if not os.path.exists(fallback):
        raise FileNotFoundError(
            f"No scaler.pkl found at {scaler_path} and no fallback training CSV at "
            f"{fallback}. Pass --scaler-path or --scaler-fit-csv."
        )
    from sklearn.preprocessing import StandardScaler
    print(f"  No scaler.pkl found; fitting one on {fallback} (matches training preprocessing).")
    train_df, _, _, train_feat_cols = load_input(fallback)
    assert train_feat_cols == feat_cols or set(train_feat_cols) == set(feat_cols), \
        "Fallback scaler-fit CSV has different feature columns than --input-csv."
    scaler = StandardScaler().fit(train_df[feat_cols].to_numpy(np.float32))
    return scaler


def compute_metrics(y_true, y_pred):
    from sklearn.metrics import (
        r2_score, mean_squared_error, accuracy_score,
        precision_score, recall_score, f1_score,
        roc_auc_score, matthews_corrcoef,
    )
    from scipy.stats import pearsonr

    yb = (y_true >= 0.5).astype(int)
    pb = (y_pred >= THRESHOLD).astype(int)
    pr, _ = pearsonr(y_true, y_pred)
    return dict(
        R2        = r2_score(y_true, y_pred),
        Pearson   = pr,
        RMSE      = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        ACC       = accuracy_score(yb, pb),
        Precision = precision_score(yb, pb, zero_division=0),
        Recall    = recall_score(yb, pb, zero_division=0),
        F1        = f1_score(yb, pb, zero_division=0),
        AUC       = roc_auc_score(yb, y_pred),
        MCC       = matthews_corrcoef(yb, pb),
    )


def predict_xgboost(args, X_raw):
    import xgboost as xgb

    preds = []
    for seed in args.seeds:
        path = os.path.join(args.model_dir, "xgboost_models", f"seed_{seed}.json")
        model = xgb.XGBRegressor()
        model.load_model(path)
        preds.append(np.clip(model.predict(X_raw), 0, 1))
        del model
    return np.mean(preds, axis=0)


def predict_torch(args, model_type, X_scaled, E, batch_size=16):
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    phys_t = torch.tensor(X_scaled, dtype=torch.float32) if X_scaled is not None else None
    esm_t = torch.tensor(E, dtype=torch.float32) if E is not None else None
    n = len(phys_t) if phys_t is not None else len(esm_t)

    def build():
        if model_type == "graph_only":
            return GraphOnlyNN(X_scaled.shape[1], dropout=0.0)
        if model_type == "esm_only":
            return ESMOnlyNN(dropout=0.0)
        if model_type == "gated_hybrid":
            return GatedHybridNN(X_scaled.shape[1], dropout=0.0)
        raise ValueError(model_type)

    all_preds = []
    for seed in args.seeds:
        ckpt = os.path.join(args.model_dir, model_type, f"model_seed{seed}_refit.pt")
        model = build().to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device))
        model.eval()

        preds = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                p = phys_t[i:i+batch_size].to(device) if phys_t is not None else None
                e = esm_t[i:i+batch_size].to(device) if esm_t is not None else None
                if model_type == "graph_only":
                    out, _ = model(None, p)
                elif model_type == "esm_only":
                    out, _ = model(e, None)
                else:  # gated_hybrid
                    out, _ = model(e, p)
                preds.append(out.cpu().numpy())
        all_preds.append(np.clip(np.concatenate(preds).ravel(), 0, 1))
        del model
    return np.mean(all_preds, axis=0)


def load_ensemble_alpha(args):
    if args.alpha is not None:
        return args.alpha
    path = os.path.join(args.model_dir, "ensemble", "metrics_per_seed.csv")
    if os.path.exists(path):
        alphas = pd.read_csv(path)["alpha"].astype(float)
        alpha = float(alphas.mean())
        print(f"  alpha={alpha:.4f} (mean of {len(alphas)} per-seed values from {path})")
        return alpha
    raise FileNotFoundError(
        f"No {path} to read alpha from, and --alpha not given. "
        "Pass --alpha explicitly (graph_only weight, 0-1)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="BioGraphX-Sol unified inference across all five trained configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-type", required=True,
                        choices=["xgboost", "graph_only", "esm_only", "gated_hybrid", "ensemble"])
    parser.add_argument("--input-csv", required=True,
                        help="Pre-encoded CSV from BioGraphX-Sol-Encoding/run.py (200 feature columns).")
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--model-dir", required=True,
                        help="Root folder with xgboost_models/, graph_only/, esm_only/, gated_hybrid/, ensemble/ "
                             "(the layout of this repo's 'Models Weights/').")
    parser.add_argument("--esm-dir", default=None,
                        help="Directory of cached {ID}.npz ESM-2 embeddings from esm_embeddings.py "
                             "(required for esm_only/gated_hybrid/ensemble).")
    parser.add_argument("--scaler-path", default=None,
                        help="StandardScaler .pkl saved during training. Defaults to <model-dir>/scaler.pkl, "
                             "falling back to fitting one on --scaler-fit-csv if that file is missing.")
    parser.add_argument("--scaler-fit-csv", default=None,
                        help="Training-encoded CSV to fit a scaler on if no scaler.pkl is found "
                             "(default: eSol_train_encoded.csv in the current directory).")
    parser.add_argument("--alpha", type=float, default=None,
                        help="graph_only weight for --model-type ensemble (0-1). Default: mean of the "
                             "per-seed alphas in <model-dir>/ensemble/metrics_per_seed.csv.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval", action="store_true",
                        help="If the input CSV has a Label/solubility column, compute and print/save metrics.")
    args = parser.parse_args()

    needs_esm = args.model_type in ("esm_only", "gated_hybrid", "ensemble")
    if needs_esm and not args.esm_dir:
        parser.error(f"--esm-dir is required for --model-type {args.model_type}")

    df, id_col, tgt_col, feat_cols = load_input(args.input_csv)
    ids = df[id_col].astype(str).tolist()
    print(f"Loaded {len(df)} protein(s) from {args.input_csv} (ID column: '{id_col}')")

    X_raw = df[feat_cols].to_numpy(np.float32)

    needs_scaled = args.model_type in ("graph_only", "gated_hybrid", "ensemble")
    X_scaled = None
    if needs_scaled:
        scaler = get_scaler(args, feat_cols)
        X_scaled = scaler.transform(X_raw)

    E = None
    if needs_esm:
        print(f"Loading ESM-2 embeddings from {args.esm_dir} ...")
        E = build_esm_matrix(ids, args.esm_dir)

    if args.model_type == "xgboost":
        preds = predict_xgboost(args, X_raw)
    elif args.model_type == "ensemble":
        alpha = load_ensemble_alpha(args)
        graph_preds = predict_torch(args, "graph_only", X_scaled, None, args.batch_size)
        esm_preds = predict_torch(args, "esm_only", X_scaled, E, args.batch_size)
        preds = np.clip(alpha * graph_preds + (1 - alpha) * esm_preds, 0, 1)
    else:
        preds = predict_torch(args, args.model_type, X_scaled, E, args.batch_size)

    out = pd.DataFrame({"ID": ids, "Predicted_Solubility": preds})

    if args.eval:
        if tgt_col is None:
            print("[warn] --eval given but no Label/solubility column found in the input CSV; skipping.")
        else:
            y_true = df[tgt_col].to_numpy(np.float32)
            m = compute_metrics(y_true, preds)
            print(f"\n{args.model_type} on {args.input_csv} ({len(df)} proteins):")
            for k, v in m.items():
                print(f"  {k:<10} {v:.4f}")
            metrics_csv = os.path.splitext(args.output_csv)[0] + "_metrics.csv"
            pd.DataFrame([m]).to_csv(metrics_csv, index=False)
            print(f"Metrics saved to {metrics_csv}")
            out["y_true"] = y_true

    out.to_csv(args.output_csv, index=False)
    print(f"Saved predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
