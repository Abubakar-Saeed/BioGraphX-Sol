#!/usr/bin/env python3
"""
BioGraphX-Sol - Graph-only NN + ESM-only NN post-hoc ensemble.

Loads already-trained graph_only and esm_only NN checkpoints (no
retraining). Protocol:

  1. For each of the 5 seeds, predict with that seed's graph_only and
     ESM-only checkpoints on the full 660-sequence test set.
  2. Per seed, grid-search alpha in {0.00, 0.05, ..., 1.00} (the weight on
     graph_only; 1 - alpha on the ESM-NN) to maximize validation R^2 on the
     268-sequence validation subset (the first 268 rows of the test set).
  3. Form that seed's ensemble prediction with its own alpha_s, evaluate on
     the full 660-sequence test set.
  4. Report mean +/- SD across the 5 per-seed ensemble predictions.

`train_nn_esol.py` also computes a graph_only+esm_only ensemble during
training, but selects alpha on the *training* set's out-of-fold
predictions - a different (and also valid) protocol. This script instead
selects alpha on a held-out slice of the test set and needs no retraining,
just the already-saved checkpoints.

`alpha_sweep.csv` additionally reports, for every alpha in the grid, the
mean +/- SD (across the 5 seeds) of validation R^2 and full-test metrics.

    graph_only : 200 physics features, scaled with the training StandardScaler
    ESM-NN     : ESM-2 embeddings only (2560-dim, no physics features)

Requires
--------
    <graph-model-dir>/model_seed<seed>_refit.pt   (from train_nn_esol.py, graph_only/)
    <esm-model-dir>/model_seed<seed>_refit.pt     (from train_nn_esol.py, esm_only/)
    --scaler-path                                 (scaler.pkl saved by train_nn_esol.py)
    --test-csv        (200 features, e.g. eSol_test_encoded.csv)
    --esm-dirs        (.npz per-residue embeddings for the test set)

Usage
-----
    python alpha_sweep_graph_esm.py \\
        --test-csv "Encoded Data/eSol_test_encoded.csv" \\
        --graph-model-dir biographx_sol_nn_results/graph_only \\
        --esm-model-dir biographx_sol_nn_results/esm_only \\
        --scaler-path biographx_sol_nn_results/scaler.pkl \\
        --esm-dirs esm_embeddings/eSol_test \\
        --output-dir graph_esm_ensemble
"""

import argparse
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train_nn_esol import (  # noqa: E402
    GraphOnlyNN, ESMOnlyNN, META_COLS, ID_CANDIDATES, TARGET_CANDIDATES,
    pick_col, build_esm_index, build_esm_matrix, compute_metrics, write_model_summary,
)

KEYS = ["R2", "Pearson", "RMSE", "ACC", "Precision", "Recall", "F1", "AUC", "MCC"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def predict_graph_nn(model, phys_mat, batch_size=32):
    model.eval()
    t = torch.tensor(phys_mat, dtype=torch.float32)
    out = []
    for i in range(0, len(t), batch_size):
        pred, _ = model(None, t[i:i+batch_size].to(DEVICE))
        out.append(pred.cpu().numpy())
    return np.concatenate(out).ravel()


@torch.no_grad()
def predict_esm_nn(model, esm_mat, batch_size=32):
    model.eval()
    t = torch.tensor(esm_mat, dtype=torch.float32)
    out = []
    for i in range(0, len(t), batch_size):
        pred, _ = model(t[i:i+batch_size].to(DEVICE), None)
        out.append(pred.cpu().numpy())
    return np.concatenate(out).ravel()


def best_alpha_for_seed(a_pred, b_pred, y_val, val_size, alpha_steps):
    best_a, best_r2 = 0.0, -np.inf
    for a in alpha_steps:
        r2v = r2_score(y_val, a * a_pred[:val_size] + (1 - a) * b_pred[:val_size])
        if r2v > best_r2:
            best_r2, best_a = r2v, float(a)
    return best_a


def run(args):
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device : {DEVICE}\n")

    test_df = pd.read_csv(args.test_csv)
    id_col = pick_col(test_df.columns, ID_CANDIDATES, "ID")
    tgt_col = pick_col(test_df.columns, TARGET_CANDIDATES, "target")
    feat_cols = [c for c in test_df.columns
                 if c not in META_COLS and c not in (id_col, tgt_col)
                 and np.issubdtype(test_df[c].dtype, np.number)]
    assert len(feat_cols) == 200, f"Expected 200 features, got {len(feat_cols)}"
    if args.expected_test_rows:
        assert len(test_df) == args.expected_test_rows, \
            f"expected {args.expected_test_rows} test rows, got {len(test_df)}"

    scaler = joblib.load(args.scaler_path)
    X_test_scaled = scaler.transform(test_df[feat_cols].to_numpy(np.float32))
    y_test = test_df[tgt_col].to_numpy(np.float32)
    y_val = y_test[:args.val_size]
    phys_dim = len(feat_cols)

    print(f"Test     : {len(test_df)}")
    print(f"Val      : {args.val_size}  (first {args.val_size} of test - alpha selection only)")
    print(f"Features : {len(feat_cols)}\n")

    esm_index = build_esm_index(args.esm_dirs)
    esm_mat = build_esm_matrix(test_df[id_col].astype(str), esm_index)

    graph_preds_all, esm_preds_all = [], []
    for seed in args.seeds:
        graph_path = os.path.join(args.graph_model_dir, f"model_seed{seed}_refit.pt")
        graph_model = GraphOnlyNN(phys_dim).to(DEVICE)
        graph_model.load_state_dict(torch.load(graph_path, map_location=DEVICE))
        graph_pred = np.clip(predict_graph_nn(graph_model, X_test_scaled, args.batch_size), 0, 1)
        graph_preds_all.append(graph_pred)
        del graph_model

        esm_path = os.path.join(args.esm_model_dir, f"model_seed{seed}_refit.pt")
        esm_model = ESMOnlyNN().to(DEVICE)
        esm_model.load_state_dict(torch.load(esm_path, map_location=DEVICE))
        esm_pred = np.clip(predict_esm_nn(esm_model, esm_mat, args.batch_size), 0, 1)
        esm_preds_all.append(esm_pred)
        del esm_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        m_graph, m_esm = compute_metrics(y_test, graph_pred), compute_metrics(y_test, esm_pred)
        print(f"  seed {seed}  Graph R2={m_graph['R2']:.4f} AUC={m_graph['AUC']:.4f}  |  "
              f"ESM-NN R2={m_esm['R2']:.4f} AUC={m_esm['AUC']:.4f}")

    # --- alpha_sweep.csv: mean +/- SD across seeds, for every alpha in the grid ---
    sweep_rows = []
    for a in args.alpha_steps:
        val_r2s, test_metrics = [], []
        for gp, ep in zip(graph_preds_all, esm_preds_all):
            blend = np.clip(a * gp + (1 - a) * ep, 0, 1)
            val_r2s.append(r2_score(y_val, blend[:args.val_size]))
            test_metrics.append(compute_metrics(y_test, blend))
        row = {"alpha": round(float(a), 2),
               "val_R2_mean": np.mean(val_r2s), "val_R2_sd": np.std(val_r2s, ddof=1)}
        for k in KEYS:
            vals = [m[k] for m in test_metrics]
            row[f"test_{k}_mean"] = np.mean(vals)
            row[f"test_{k}_sd"] = np.std(vals, ddof=1)
        sweep_rows.append(row)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(args.output_dir, "alpha_sweep.csv"), index=False)

    # --- per-seed alpha selection + per-seed ensemble ---
    per_seed_rows = []
    for seed, gp, ep in zip(args.seeds, graph_preds_all, esm_preds_all):
        alpha_s = best_alpha_for_seed(gp, ep, y_val, args.val_size, args.alpha_steps)
        blend_s = np.clip(alpha_s * gp + (1 - alpha_s) * ep, 0, 1)
        m_s = compute_metrics(y_test, blend_s)
        per_seed_rows.append({"seed": seed, "alpha": alpha_s, **m_s})
        print(f"  seed {seed}  alpha={alpha_s:.2f}  R2={m_s['R2']:.4f}  AUC={m_s['AUC']:.4f}")

    mean, sd = write_model_summary(args.output_dir, per_seed_rows)
    alpha_mean = np.mean([r["alpha"] for r in per_seed_rows])
    alpha_sd = np.std([r["alpha"] for r in per_seed_rows], ddof=1)

    print(f"\ngraph_only+ESM ensemble ({len(args.seeds)} seeds):")
    print(f"  alpha = {alpha_mean:.2f} +/- {alpha_sd:.2f}  "
          f"(graph_only {alpha_mean*100:.0f}% / ESM-NN {(1-alpha_mean)*100:.0f}%)")
    for k in KEYS:
        print(f"  {k:<10} {mean[k]:.4f} +/- {sd[k]:.4f}")

    print(f"\nDone. -> {args.output_dir}/  (alpha_sweep.csv, metrics_per_seed.csv, summary.csv)")


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc graph_only + ESM-only NN ensemble (per-seed alpha selection, no retraining).")
    parser.add_argument("--test-csv", default="eSol_test_encoded.csv")
    parser.add_argument("--graph-model-dir", required=True,
                        help="Directory with model_seed<seed>_refit.pt graph_only checkpoints.")
    parser.add_argument("--esm-model-dir", required=True,
                        help="Directory with model_seed<seed>_refit.pt ESM-only checkpoints.")
    parser.add_argument("--scaler-path", required=True, help="scaler.pkl saved by train_nn_esol.py.")
    parser.add_argument("--esm-dirs", nargs="+", required=True, help="Directories of {id}.npz ESM-2 embeddings.")
    parser.add_argument("--output-dir", default="graph_esm_ensemble")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--val-size", type=int, default=268,
                         help="Leading rows of --test-csv held out for alpha selection (default: 268).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--expected-test-rows", type=int, default=660,
                        help="Sanity check on the test CSV row count. Pass 0 to disable.")
    args = parser.parse_args()
    args.alpha_steps = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    if args.expected_test_rows == 0:
        args.expected_test_rows = None
    run(args)


if __name__ == "__main__":
    main()
