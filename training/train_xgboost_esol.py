#!/usr/bin/env python3
"""
BioGraphX-Sol - 5-Seed XGBoost Training (ProtSATT Protocol)

- Train: 2019 sequences (eSOL train split)
- Test:  660 sequences (full eSOL test CSV, no val split)
- Early stopping: 5-fold CV inside the 2019 train -> mean best n_estimators
- Refit: full 2019 with that fixed n_estimators
- Saves model for each seed, per-seed and aggregated feature importance

Protocol reference: ProtSATT (Deng et al., 2026) - trained and evaluated on
the same 2019/660 eSOL split.

Usage
-----
    python train_xgboost_esol.py \\
        --train-csv eSol_train_encoded.csv \\
        --test-csv eSol_test_encoded.csv \\
        --model-dir models_protsatt \\
        --importance-dir importance_protsatt
"""

import argparse
import os

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef,
)
from scipy.stats import pearsonr

# Fixed base params - n_estimators determined by CV per seed
XGBOOST_PARAMS = dict(
    learning_rate    = 0.01,
    max_depth        = 5,
    subsample        = 0.8,
    colsample_bytree = 0.7,
    n_jobs           = -1,
    verbosity        = 0,
)

KEYS = ["R2", "Pearson", "RMSE", "ACC", "Precision", "Recall", "F1", "AUC", "MCC"]


def evaluate(preds, y):
    yb = (y >= 0.5).astype(int)
    pb = (preds >= 0.5).astype(int)
    pr, _ = pearsonr(y, preds)
    return dict(
        R2        = r2_score(y, preds),
        Pearson   = pr,
        RMSE      = float(np.sqrt(mean_squared_error(y, preds))),
        ACC       = accuracy_score(yb, pb),
        Precision = precision_score(yb, pb, zero_division=0),
        Recall    = recall_score(yb, pb, zero_division=0),
        F1        = f1_score(yb, pb, zero_division=0),
        AUC       = roc_auc_score(yb, preds),
        MCC       = matthews_corrcoef(yb, pb),
    )


def run(args):
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.importance_dir, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    test_df  = pd.read_csv(args.test_csv)

    target = "Label" if "Label" in train_df.columns else "solubility"
    excl = {"gene", target, "sequence", "pdb_filename", "ID", "uniprot"}
    feats = [c for c in train_df.columns if c not in excl]

    assert len(feats) == 200, f"Expected 200 features, got {len(feats)}"
    if args.expected_test_rows is not None:
        assert len(test_df) == args.expected_test_rows, \
            f"Expected {args.expected_test_rows} test rows, got {len(test_df)}"

    print(f"Features : {len(feats)}")
    print(f"Train    : {len(train_df)}")
    print(f"Test     : {len(test_df)}  (ProtSATT protocol)\n")

    X_train = train_df[feats].values
    y_train = train_df[target].values
    X_test  = test_df[feats].values
    y_test  = test_df[target].values

    cv_params = dict(**XGBOOST_PARAMS, n_estimators=1000, early_stopping_rounds=args.early_stopping_rounds)

    all_metrics = {k: [] for k in KEYS}
    all_gain, all_weight, all_cover = [], [], []
    cv_epoch_records = []

    for seed in args.seeds:
        print(f"Seed {seed}")

        # Step 1: 5-fold CV inside training to find best n_estimators
        kf = KFold(n_splits=args.cv_folds, shuffle=True, random_state=seed)
        best_iters = []
        for fold, (tr_idx, va_idx) in enumerate(kf.split(X_train)):
            Xtr, Xva = X_train[tr_idx], X_train[va_idx]
            ytr, yva = y_train[tr_idx], y_train[va_idx]
            cv_model = xgb.XGBRegressor(**cv_params, random_state=seed + fold)
            cv_model.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
            best_iters.append(cv_model.best_iteration + 1)
            del cv_model

        mean_iters = int(round(np.mean(best_iters)))
        print(f"  CV fold best iters: {best_iters} -> refit with n_estimators={mean_iters}")
        cv_epoch_records.append({"seed": seed, "fold_iters": best_iters, "mean_iters": mean_iters})

        # Step 2: Refit on full train with fixed n_estimators
        refit_model = xgb.XGBRegressor(**XGBOOST_PARAMS, n_estimators=mean_iters, random_state=seed)
        refit_model.fit(X_train, y_train, verbose=False)

        # Step 3: Evaluate on full test set
        preds = refit_model.predict(X_test)
        m = evaluate(preds, y_test)
        for k, v in m.items():
            all_metrics[k].append(v)
        print(f"  R2={m['R2']:.4f}  AUC={m['AUC']:.4f}  MCC={m['MCC']:.4f}  Rec={m['Recall']:.4f}")

        refit_model.save_model(os.path.join(args.model_dir, f"seed_{seed}.json"))

        booster = refit_model.get_booster()

        def get_scores(importance_type):
            raw = booster.get_score(importance_type=importance_type)
            scores = {feats[int(k[1:])]: v for k, v in raw.items()}
            return {f: scores.get(f, 0.0) for f in feats}

        gain_scores = get_scores("gain")
        weight_scores = get_scores("weight")
        cover_scores = get_scores("cover")
        all_gain.append(gain_scores)
        all_weight.append(weight_scores)
        all_cover.append(cover_scores)

        seed_df = (pd.DataFrame({
            "feature": feats,
            "gain":    [gain_scores[f]   for f in feats],
            "weight":  [weight_scores[f] for f in feats],
            "cover":   [cover_scores[f]  for f in feats],
        }).sort_values("gain", ascending=False).reset_index(drop=True))
        seed_df.index += 1
        seed_df.to_csv(os.path.join(args.importance_dir, f"importance_seed_{seed}.csv"))

        del refit_model
        print()

    cv_df = pd.DataFrame(cv_epoch_records)
    cv_df.to_csv(os.path.join(args.importance_dir, "cv_epochs.csv"), index=False)

    agg_rows = []
    for f in feats:
        gains = [d[f] for d in all_gain]
        weights = [d[f] for d in all_weight]
        covers = [d[f] for d in all_cover]
        agg_rows.append(dict(
            feature=f,
            gain_mean=np.mean(gains), gain_std=np.std(gains),
            gain_min=np.min(gains), gain_max=np.max(gains),
            weight_mean=np.mean(weights), weight_std=np.std(weights),
            cover_mean=np.mean(covers), cover_std=np.std(covers),
        ))
    agg_df = (pd.DataFrame(agg_rows).sort_values("gain_mean", ascending=False).reset_index(drop=True))
    agg_df.index += 1
    agg_df.to_csv(os.path.join(args.importance_dir, "importance_aggregated.csv"))

    means = {k: np.mean(v) for k, v in all_metrics.items()}
    stds  = {k: np.std(v)  for k, v in all_metrics.items()}

    w = 28
    print("\n" + "="*105)
    print("BioGraphX-Sol - XGBoost, ProtSATT protocol (5 seeds)")
    print("="*105)
    print(f"  {'Model':<{w}}  " + "  ".join(f"{k:>8}" for k in KEYS))
    print("  " + "-"*105)

    row = f"  {'BioGraphX-Sol (XGBoost)':<{w}}  "
    for k in KEYS:
        row += f"  {means[k]:>5.4f}+/-{stds[k]:.3f}"
    print(row)
    print("  " + "-"*105)

    print(f"\n  Per-seed (test):")
    print(f"  {'Seed':<8}  {'R2':>7}  {'AUC':>7}  {'MCC':>7}  {'Recall':>7}")
    print("  " + "-"*38)
    for i, seed in enumerate(args.seeds):
        print(f"  {seed:<8}  {all_metrics['R2'][i]:>7.4f}  {all_metrics['AUC'][i]:>7.4f}  "
              f"{all_metrics['MCC'][i]:>7.4f}  {all_metrics['Recall'][i]:>7.4f}")

    print(f"\n  Mean +/- SD (test):")
    for k in ["R2", "AUC", "MCC", "Recall"]:
        print(f"    {k:<10}  {means[k]:.4f} +/- {stds[k]:.4f}")

    print(f"\n  Top-20 Features by Mean Gain:")
    print(f"  {'Rank':<5}  {'Feature':<50}  {'Gain Mean':>10}  {'Gain Std':>9}")
    print("  " + "-"*78)
    for rank, row in agg_df.head(20).iterrows():
        print(f"  {rank:<5}  {row['feature']:<50}  {row['gain_mean']:>10.3f}  {row['gain_std']:>9.3f}")

    print(f"\nDone.")
    print(f"    Models     -> {args.model_dir}/seed_{{seed}}.json")
    print(f"    Importance -> {args.importance_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Train the 5-seed XGBoost baseline on 200 BioGraphX-Sol features.")
    parser.add_argument("--train-csv", default="eSol_train_encoded.csv")
    parser.add_argument("--test-csv", default="eSol_test_encoded.csv")
    parser.add_argument("--model-dir", default="models_protsatt")
    parser.add_argument("--importance-dir", default="importance_protsatt")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--expected-test-rows", type=int, default=660,
                        help="Sanity check on the test CSV row count. Pass 0 to disable.")
    args = parser.parse_args()
    if args.expected_test_rows == 0:
        args.expected_test_rows = None
    run(args)


if __name__ == "__main__":
    main()
