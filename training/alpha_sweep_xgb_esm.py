#!/usr/bin/env python3
"""
BioGraphX-Sol - XGBoost + ESM-only NN post-hoc ensemble.

Loads already-trained XGBoost and ESM-only NN checkpoints (no retraining).
Protocol:

  1. For each of the 5 seeds, predict with that seed's XGBoost and ESM-only
     checkpoints on the full 660-sequence test set.
  2. Per seed, grid-search alpha in {0.00, 0.05, ..., 1.00} (the weight on
     XGBoost; 1 - alpha on the ESM-NN) to maximize validation R^2 on the
     268-sequence validation subset (the first 268 rows of the test set).
  3. Form that seed's ensemble prediction with its own alpha_s, evaluate on
     the full 660-sequence test set.
  4. Report mean +/- SD across the 5 per-seed ensemble predictions - so both
     model-training variability and alpha-selection variability are
     reflected in the reported standard deviation, exactly as for every
     other configuration in this repo.

`alpha_sweep.csv` additionally reports, for every alpha in the grid, the
mean +/- SD (across the 5 seeds) of validation R^2 and full-test metrics,
not itself used to pick alpha_s (that selection is per-seed, on validation
R^2 alone).

    XGBoost : raw unscaled features (matches XGB training - no StandardScaler)
    ESM-NN  : ESM-2 embeddings only (2560-dim, no physics features)

Requires
--------
    <xgb-model-dir>/seed_<seed>.json                     (from train_xgboost_esol.py)
    <esm-model-dir>/model_seed<seed>_refit.pt             (from train_nn_esol.py, esm_only/)
    --test-csv        (200 features, e.g. eSol_test_encoded.csv)
    --esm-dirs        (.npz per-residue embeddings for the test set)

Usage
-----
    python alpha_sweep_xgb_esm.py \\
        --test-csv "Encoded Data/eSol_test_encoded.csv" \\
        --xgb-model-dir models_protsatt \\
        --esm-model-dir biographx_sol_nn_results/esm_only \\
        --esm-dirs esm_embeddings/eSol_test \\
        --output-dir xgb_esm_ensemble
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train_nn_esol import (  # noqa: E402
    ESMOnlyNN, META_COLS, ID_CANDIDATES, TARGET_CANDIDATES,
    pick_col, build_esm_index, build_esm_matrix, compute_metrics, write_model_summary,
)

KEYS = ["R2", "Pearson", "RMSE", "ACC", "Precision", "Recall", "F1", "AUC", "MCC"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

    X_test_raw = test_df[feat_cols].to_numpy(np.float32)   # unscaled - matches XGB training
    y_test = test_df[tgt_col].to_numpy(np.float32)
    y_val = y_test[:args.val_size]

    print(f"Test     : {len(test_df)}")
    print(f"Val      : {args.val_size}  (first {args.val_size} of test - alpha selection only)")
    print(f"Features : {len(feat_cols)}\n")

    esm_index = build_esm_index(args.esm_dirs)
    esm_mat = build_esm_matrix(test_df[id_col].astype(str), esm_index)

    xgb_preds_all, esm_preds_all = [], []
    for seed in args.seeds:
        xgb_path = os.path.join(args.xgb_model_dir, f"seed_{seed}.json")
        xgb_model = xgb.XGBRegressor()
        xgb_model.load_model(xgb_path)
        xgb_pred = np.clip(xgb_model.predict(X_test_raw), 0, 1)
        xgb_preds_all.append(xgb_pred)
        del xgb_model

        esm_path = os.path.join(args.esm_model_dir, f"model_seed{seed}_refit.pt")
        esm_model = ESMOnlyNN().to(DEVICE)
        esm_model.load_state_dict(torch.load(esm_path, map_location=DEVICE))
        esm_pred = np.clip(predict_esm_nn(esm_model, esm_mat, args.batch_size), 0, 1)
        esm_preds_all.append(esm_pred)
        del esm_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()

        m_xgb, m_esm = compute_metrics(y_test, xgb_pred), compute_metrics(y_test, esm_pred)
        print(f"  seed {seed}  XGB R2={m_xgb['R2']:.4f} AUC={m_xgb['AUC']:.4f}  |  "
              f"ESM-NN R2={m_esm['R2']:.4f} AUC={m_esm['AUC']:.4f}")

    # --- alpha_sweep.csv: mean +/- SD across seeds, for every alpha in the grid ---
    sweep_rows = []
    for a in args.alpha_steps:
        val_r2s, test_metrics = [], []
        for xp, ep in zip(xgb_preds_all, esm_preds_all):
            blend = np.clip(a * xp + (1 - a) * ep, 0, 1)
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
    for seed, xp, ep in zip(args.seeds, xgb_preds_all, esm_preds_all):
        alpha_s = best_alpha_for_seed(xp, ep, y_val, args.val_size, args.alpha_steps)
        blend_s = np.clip(alpha_s * xp + (1 - alpha_s) * ep, 0, 1)
        m_s = compute_metrics(y_test, blend_s)
        per_seed_rows.append({"seed": seed, "alpha": alpha_s, **m_s})
        print(f"  seed {seed}  alpha={alpha_s:.2f}  R2={m_s['R2']:.4f}  AUC={m_s['AUC']:.4f}")

    mean, sd = write_model_summary(args.output_dir, per_seed_rows)
    alpha_mean = np.mean([r["alpha"] for r in per_seed_rows])
    alpha_sd = np.std([r["alpha"] for r in per_seed_rows], ddof=1)

    print(f"\nXGB+ESM ensemble ({len(args.seeds)} seeds):")
    print(f"  alpha = {alpha_mean:.2f} +/- {alpha_sd:.2f}  "
          f"(XGBoost {alpha_mean*100:.0f}% / ESM-NN {(1-alpha_mean)*100:.0f}%)")
    for k in KEYS:
        print(f"  {k:<10} {mean[k]:.4f} +/- {sd[k]:.4f}")

    print(f"\nDone. -> {args.output_dir}/  (alpha_sweep.csv, metrics_per_seed.csv, summary.csv)")


def main():
    parser = argparse.ArgumentParser(
        description="Post-hoc XGBoost + ESM-only NN ensemble (per-seed alpha selection, no retraining).")
    parser.add_argument("--test-csv", default="eSol_test_encoded.csv")
    parser.add_argument("--xgb-model-dir", required=True, help="Directory with seed_<seed>.json checkpoints.")
    parser.add_argument("--esm-model-dir", required=True,
                        help="Directory with model_seed<seed>_refit.pt ESM-only checkpoints.")
    parser.add_argument("--esm-dirs", nargs="+", required=True, help="Directories of {id}.npz ESM-2 embeddings.")
    parser.add_argument("--output-dir", default="xgb_esm_ensemble")
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
