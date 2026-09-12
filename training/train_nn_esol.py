#!/usr/bin/env python3
"""
BioGraphX-Sol - Neural Network Training on eSOL (ProtSATT protocol).

Trains all three base models together, sharing the same CV folds, because
the post-hoc ensemble needs graph_only's and esm_only's out-of-fold
predictions from the SAME split:

  - Physics : 200 BioGraphX-Sol features (pre-encoded CSV)
  - ESM     : esm2_t36_3B_UR50D, 2560-dim mean-pooled per-residue embedding
4 configs : graph_only / esm_only / gated_hybrid / ensemble (post-hoc alpha blend)

This is why, unlike BioGraphX-IDP's training/ layout (one script per model
type), BioGraphX-Sol has a single `train_nn_esol.py`: splitting it into three
independent scripts would mean re-loading each other's out-of-fold
predictions from disk instead of sharing them directly in memory, changing
the actual algorithm rather than just its file layout.

Protocol (matches the ProtSATT paper: train_eSOL_fold.py -> train_eSOL.py):
  1. KFold(5, shuffle) inside the 2019-row train set.
  2. Each fold: train, pick the best epoch by validation metric (R^2 by default,
     as ProtSATT does; MSE optional). Fold models are saved.
  3. mean best epoch over the 5 folds -> refit on the FULL 2019 train for that
     many epochs (no early stop). Refit model saved.
  4. Evaluate once on the FULL 660-row test set.
  5. Ensemble: alpha (graph vs ESM) chosen on the out-of-fold predictions over
     the 2019 train, reported on the 660 test.
Repeated over 5 seeds -> mean +/- sample-SD (ddof=1).

Outputs (--output-dir/):
  scaler.pkl, esm_train.npy, esm_test.npy
  <model>/  model_seed<s>_fold<k>.pt  model_seed<s>_refit.pt
            cv_epochs.csv  metrics_per_seed.csv  summary.csv
            gate_history.csv                       (gated_hybrid only)
            alpha_sweep_seed<s>.csv                (ensemble only)
  predictions/  test_preds_seed<s>.csv  oof_preds_seed<s>.csv
  final_summary.csv        all 4 configs, mean + sd
  <output-dir>.zip         one archive of everything          (--zip-outputs)

Usage
-----
    python train_nn_esol.py \\
        --train-csv eSol_train_encoded.csv \\
        --test-csv eSol_test_encoded.csv \\
        --esm-dirs esm_embeddings/eSol_train esm_embeddings/eSol_test \\
        --output-dir biographx_sol_nn_results
"""

import argparse
import os
import gc
import glob
import shutil
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    r2_score, mean_squared_error, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, matthews_corrcoef,
)
from scipy.stats import pearsonr
import joblib

warnings.filterwarnings("ignore")

META_COLS         = {"ID", "gene", "uniprot", "sequence", "pdb_filename",
                     "Label", "solubility", "target"}
ID_CANDIDATES     = ["gene", "uniprot", "ID"]
TARGET_CANDIDATES = ["Label", "solubility", "target"]
THRESHOLD         = 0.5

# Architecture
ESM_DIM    = 2560            # esm2_t36_3B_UR50D
SHARED_DIM = 512
HIDDEN_DIM = 256

KEYS = ["R2", "Pearson", "RMSE", "ACC", "Precision", "Recall", "F1", "AUC", "MCC"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================================
# 1. DATA UTILITIES
# ============================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def pick_col(cols, candidates, what):
    hit = next((c for c in candidates if c in cols), None)
    assert hit, f"no {what} column found (looked for {candidates})"
    return hit


def build_esm_index(dir_list):
    index = {}
    for d in dir_list:
        if not os.path.isdir(d):
            print(f"  [warn] ESM dir not found: {d}")
            continue
        for f in glob.glob(os.path.join(d, "*.npz")):
            index[os.path.basename(f)[:-4]] = f
    print(f"ESM index: {len(index)} .npz embeddings")
    return index


def load_esm_embedding(path):
    """Per-residue (L, 2560) -> mean-pool over residues, dropping BOS/EOS."""
    try:
        data = np.load(path)
        key = "embedding" if "embedding" in data else data.files[0]
        emb = data[key].astype(np.float32)
        if emb.ndim == 2 and emb.shape[0] > 2:
            emb = emb[1:-1]
        return emb.mean(axis=0) if emb.ndim == 2 else emb.astype(np.float32)
    except Exception as e:                        # noqa: BLE001
        print(f"  [warn] failed to load {path}: {e}")
        return np.zeros(ESM_DIM, dtype=np.float32)


def build_esm_matrix(ids, esm_index, cache_path=None):
    """(N, ESM_DIM) mean-pooled ESM matrix aligned to `ids`. Cached to .npy."""
    if cache_path and os.path.exists(cache_path):
        mat = np.load(cache_path)
        if mat.shape == (len(ids), ESM_DIM):
            print(f"  ESM matrix from cache: {cache_path}  {mat.shape}")
            return mat
    mat = np.zeros((len(ids), ESM_DIM), dtype=np.float32)
    missing = 0
    for i, pid in enumerate(ids):
        path = esm_index.get(str(pid))
        if path is None:
            missing += 1
            continue
        mat[i] = load_esm_embedding(path)
    if missing:
        print(f"  [warn] {missing}/{len(ids)} ids had NO ESM embedding (left as zeros)")
    if cache_path:
        np.save(cache_path, mat)
    return mat


class SolubilityDataset(Dataset):
    def __init__(self, phys_mat, esm_mat, targets):
        self.phys = torch.as_tensor(phys_mat, dtype=torch.float32)
        self.esm  = torch.as_tensor(esm_mat,  dtype=torch.float32)
        self.y    = torch.as_tensor(np.asarray(targets, np.float32)).view(-1, 1)

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, i):
        return self.esm[i], self.phys[i], self.y[i]


def make_loader(ds, shuffle, batch_size, num_workers, seed=0):
    g = torch.Generator().manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=(DEVICE == "cuda"),
                      generator=g if shuffle else None)


# ============================================================================
# 2. MODELS
# ============================================================================
def _mlp_block(in_dim, out_dim, dropout):
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.BatchNorm1d(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class GraphOnlyNN(nn.Module):
    def __init__(self, phys_dim, dropout=0.3):
        super().__init__()
        self.phys_branch = nn.Sequential(
            _mlp_block(phys_dim, SHARED_DIM, dropout),
            _mlp_block(SHARED_DIM, HIDDEN_DIM, dropout),
        )
        self.head = nn.Linear(HIDDEN_DIM, 1)

    def forward(self, esm, phys):
        return torch.sigmoid(self.head(self.phys_branch(phys))), None


class ESMOnlyNN(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            _mlp_block(ESM_DIM, SHARED_DIM, dropout),
            _mlp_block(SHARED_DIM, HIDDEN_DIM, dropout),
            nn.Linear(HIDDEN_DIM, 1),
        )

    def forward(self, esm, phys):
        return torch.sigmoid(self.net(esm)), None


class GatedHybridNN(nn.Module):
    """Gated fusion - mirrors BioGraphX-Net. Physics branch gets more capacity
    to offset the 2560-vs-200 dimensionality asymmetry."""
    def __init__(self, phys_dim, dropout=0.3):
        super().__init__()
        self.phys_branch = nn.Sequential(
            _mlp_block(phys_dim,   SHARED_DIM, dropout),
            _mlp_block(SHARED_DIM, SHARED_DIM, dropout),
            _mlp_block(SHARED_DIM, SHARED_DIM, dropout),
        )
        self.esm_branch = nn.Sequential(_mlp_block(ESM_DIM, SHARED_DIM, dropout))
        self.gate = nn.Sequential(
            nn.Linear(SHARED_DIM * 2, HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(HIDDEN_DIM, 2),
            nn.Sigmoid(),                        # [g_physics, g_esm]
        )
        self.head = nn.Sequential(
            _mlp_block(SHARED_DIM * 2, SHARED_DIM, dropout),
            nn.Linear(SHARED_DIM, 1),
        )

    def forward(self, esm, phys):
        Bf = self.phys_branch(phys)
        Ef = self.esm_branch(esm)
        gates = self.gate(torch.cat([Ef, Bf], dim=1))
        g_phys, g_esm = gates[:, 0:1], gates[:, 1:2]
        gated = torch.cat([Ef * g_esm, Bf * g_phys], dim=1)
        return torch.sigmoid(self.head(gated)), gates


MODEL_BUILDERS = {
    "graph_only":   lambda phys_dim, dropout: GraphOnlyNN(phys_dim, dropout),
    "esm_only":     lambda phys_dim, dropout: ESMOnlyNN(dropout),
    "gated_hybrid": lambda phys_dim, dropout: GatedHybridNN(phys_dim, dropout),
}
BASE_MODELS = ["graph_only", "esm_only", "gated_hybrid"]


# ============================================================================
# 3. METRICS
# ============================================================================
def compute_metrics(y_true, y_pred):
    y_true = np.asarray(y_true, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    yb = (y_true >= 0.5).astype(int)
    pb = (y_pred >= THRESHOLD).astype(int)
    try:
        auc = roc_auc_score(yb, np.clip(y_pred, 0, 1)) if yb.min() != yb.max() else np.nan
    except ValueError:
        auc = np.nan
    return dict(
        R2        = r2_score(y_true, y_pred),
        Pearson   = pearsonr(y_true, y_pred)[0],
        RMSE      = float(np.sqrt(mean_squared_error(y_true, y_pred))),
        ACC       = accuracy_score(yb, pb),
        Precision = precision_score(yb, pb, zero_division=0),
        Recall    = recall_score(yb, pb, zero_division=0),
        F1        = f1_score(yb, pb, zero_division=0),
        AUC       = auc,
        MCC       = matthews_corrcoef(yb, pb),
    )


# ============================================================================
# 4. TRAINING
# ============================================================================
def _optimizer(model, lr_base, lr_physics, weight_decay):
    phys  = [p for n, p in model.named_parameters() if "phys_branch" in n]
    other = [p for n, p in model.named_parameters() if "phys_branch" not in n]
    groups = ([{"params": phys, "lr": lr_physics}] if phys else []) + \
             [{"params": other, "lr": lr_base}]
    return torch.optim.AdamW(groups, weight_decay=weight_decay)


def _better(new, best, epoch_select):
    return new > best if epoch_select == "r2" else new < best


def train_model(model, train_loader, val_loader, seed, tag, cfg, fixed_epochs=None):
    """
    val_loader given   -> pick best epoch by cfg.epoch_select ('r2' max / 'mse' min),
                          patience cfg.patience, restore best weights.
                          returns (model, gate_log, best_epoch).
    fixed_epochs given -> train exactly that many epochs, no validation.
    """
    set_seed(seed)
    model.to(DEVICE)
    crit = nn.MSELoss()
    opt = _optimizer(model, cfg.lr_base, cfg.lr_physics, cfg.weight_decay)
    total = fixed_epochs or cfg.max_epochs
    steps_per_epoch = max(1, len(train_loader))
    # OneCycleLR: 10% warmup (start at max_lr/10) then cosine anneal to max_lr/1000.
    # max_lr per param group so it works whether _optimizer made 1 or 2 groups.
    # Stepped per BATCH. With early stopping the cosine tail is not reached, which
    # is fine for the CV folds (they only estimate the epoch count + OOF); the
    # refit runs the full `total` epochs so it gets the complete schedule.
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in opt.param_groups],
        epochs=total, steps_per_epoch=steps_per_epoch,
        pct_start=0.1, anneal_strategy="cos",
        div_factor=10.0, final_div_factor=100.0,
    )

    best_score = -np.inf if cfg.epoch_select == "r2" else np.inf
    best_state, best_epoch, wait = None, 0, 0
    gate_log = []

    for epoch in range(total):
        model.train()
        for esm, phys, tgt in train_loader:
            esm, phys, tgt = esm.to(DEVICE), phys.to(DEVICE), tgt.to(DEVICE)
            opt.zero_grad()
            pred, gates = model(esm, phys)
            loss = crit(pred, tgt)
            if epoch < cfg.enc_epochs and gates is not None:
                loss = loss + cfg.enc_lambda * F.mse_loss(
                    gates[:, 0].mean(), torch.tensor(0.5, device=DEVICE))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_norm)
            opt.step()
            sched.step()          # OneCycleLR steps per batch

        if val_loader is None:
            continue

        model.eval()
        vp, vt, vg = [], [], []
        with torch.no_grad():
            for esm, phys, tgt in val_loader:
                esm, phys = esm.to(DEVICE), phys.to(DEVICE)
                pred, gates = model(esm, phys)
                vp.append(pred.cpu().numpy()); vt.append(tgt.numpy())
                if gates is not None:
                    vg.append(gates.cpu().numpy())
        vp = np.concatenate(vp).ravel(); vt = np.concatenate(vt).ravel()
        score = r2_score(vt, vp) if cfg.epoch_select == "r2" else mean_squared_error(vt, vp)
        if vg:
            g = np.vstack(vg)
            gate_log.append(dict(epoch=epoch, g_physics=float(g[:, 0].mean()),
                                 g_esm=float(g[:, 1].mean())))
        if _better(score, best_score, cfg.epoch_select):
            best_score, best_epoch, wait = score, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if val_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        print(f"    {tag}: best_epoch={best_epoch + 1}  val_{cfg.epoch_select}={best_score:.4f}")
        return model, gate_log, best_epoch + 1
    print(f"    {tag}: trained {total} epochs (fixed)")
    return model, gate_log, total


@torch.no_grad()
def predict(model, loader):
    model.eval()
    out = []
    for esm, phys, _ in loader:
        esm, phys = esm.to(DEVICE), phys.to(DEVICE)
        pred, _ = model(esm, phys)
        out.append(pred.cpu().numpy())
    return np.concatenate(out).ravel()


def alpha_sweep(g_sel, e_sel, y_sel, g_test, e_test, y_test, alpha_steps):
    rows, best_a, best_r2 = [], 0.0, -np.inf
    for a in alpha_steps:
        r2v = r2_score(y_sel, a * g_sel + (1 - a) * e_sel)
        rows.append(dict(alpha=float(a), r2_sel=r2v,
                         r2_test=r2_score(y_test, a * g_test + (1 - a) * e_test)))
        if r2v > best_r2:
            best_r2, best_a = r2v, float(a)
    blend = np.clip(best_a * g_test + (1 - best_a) * e_test, 0, 1)
    return best_a, compute_metrics(y_test, blend), pd.DataFrame(rows)


# ============================================================================
# 5. OUTPUT HELPERS
# ============================================================================
def mdir(output_dir, name):
    p = os.path.join(output_dir, name)
    os.makedirs(p, exist_ok=True)
    return p


def zip_dir(folder, do_zip):
    if do_zip and os.path.isdir(folder):
        shutil.make_archive(folder, "zip", root_dir=folder)


def write_model_summary(folder, per_seed_rows):
    df = pd.DataFrame(per_seed_rows)
    df.to_csv(os.path.join(folder, "metrics_per_seed.csv"), index=False)
    mean, sd = df[KEYS].mean(), df[KEYS].std(ddof=1)
    pd.DataFrame({"metric": KEYS, "mean": mean[KEYS].values, "sd": sd[KEYS].values}
                 ).to_csv(os.path.join(folder, "summary.csv"), index=False)
    return mean, sd


# ============================================================================
# 6. MAIN - ProtSATT protocol
# ============================================================================
def run(cfg):
    os.makedirs(os.path.join(cfg.output_dir, "predictions"), exist_ok=True)
    print(f"Device : {DEVICE}\nOutput : {cfg.output_dir}/\nEpoch selection : {cfg.epoch_select}\n")

    train_full = pd.read_csv(cfg.train_csv)
    test_full  = pd.read_csv(cfg.test_csv)
    id_col  = pick_col(train_full.columns, ID_CANDIDATES, "ID")
    tgt_col = pick_col(train_full.columns, TARGET_CANDIDATES, "target")
    feat_cols = [c for c in train_full.columns
                 if c not in META_COLS and c not in (id_col, tgt_col)
                 and np.issubdtype(train_full[c].dtype, np.number)]
    print(f"ID='{id_col}'  target='{tgt_col}'  features={len(feat_cols)}")
    print(f"train={len(train_full)}  test={len(test_full)}")
    if cfg.expected_test_rows:
        assert len(test_full) == cfg.expected_test_rows, \
            f"expected {cfg.expected_test_rows} test rows, got {len(test_full)}"

    esm_index = build_esm_index(cfg.esm_dirs)

    scaler = StandardScaler().fit(train_full[feat_cols].to_numpy(np.float32))
    joblib.dump(scaler, os.path.join(cfg.output_dir, "scaler.pkl"))
    Xtr = scaler.transform(train_full[feat_cols].to_numpy(np.float32))
    Xte = scaler.transform(test_full[feat_cols].to_numpy(np.float32))
    Etr = build_esm_matrix(train_full[id_col].astype(str), esm_index,
                           os.path.join(cfg.output_dir, "esm_train.npy"))
    Ete = build_esm_matrix(test_full[id_col].astype(str), esm_index,
                           os.path.join(cfg.output_dir, "esm_test.npy"))
    ytr = train_full[tgt_col].to_numpy(np.float32)
    yte = test_full[tgt_col].to_numpy(np.float32)
    phys_dim = len(feat_cols)

    ds_tr = SolubilityDataset(Xtr, Etr, ytr)
    ds_te = SolubilityDataset(Xte, Ete, yte)
    te_ld = make_loader(ds_te, shuffle=False, batch_size=cfg.batch_size, num_workers=cfg.num_workers)

    per_seed  = {m: [] for m in BASE_MODELS + ["ensemble"]}
    gate_hist = []
    cv_epoch_rows = {m: [] for m in BASE_MODELS}

    for seed in cfg.seeds:
        print(f"\n{'='*70}\n  SEED {seed}\n{'='*70}")
        folds = list(KFold(n_splits=cfg.cv_folds, shuffle=True,
                           random_state=seed).split(np.arange(len(train_full))))
        oof, test_preds = {}, {}

        for name in BASE_MODELS:
            folder = mdir(cfg.output_dir, name)
            oof_v = np.zeros(len(train_full), dtype=np.float32)
            best_epochs = []

            for k, (tr_idx, va_idx) in enumerate(folds):
                tr_ld = make_loader(Subset(ds_tr, tr_idx), shuffle=True, seed=seed + k,
                                    batch_size=cfg.batch_size, num_workers=cfg.num_workers)
                va_ld = make_loader(Subset(ds_tr, va_idx), shuffle=False,
                                    batch_size=cfg.batch_size, num_workers=cfg.num_workers)
                model = MODEL_BUILDERS[name](phys_dim, cfg.dropout)
                model, glog, best_ep = train_model(model, tr_ld, va_ld,
                                                   seed + k, f"{name} fold{k}", cfg)
                torch.save(model.state_dict(),
                           os.path.join(folder, f"model_seed{seed}_fold{k}.pt"))
                oof_v[va_idx] = predict(model, va_ld)
                best_epochs.append(best_ep)
                cv_epoch_rows[name].append({"seed": seed, "fold": k, "best_epoch": best_ep})
                if name == "gated_hybrid":
                    gate_hist += [{"seed": seed, "fold": k, **r} for r in glog]
                del model
                if DEVICE == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            # 75th percentile, not mean: fast-converging folds must not shorten
            # the refit. Use max(best_epochs) to be even more aggressive.
            refit_ep = int(round(np.percentile(best_epochs, 75)))
            print(f"    {name}: fold epochs {best_epochs} -> refit {refit_ep}")
            refit = MODEL_BUILDERS[name](phys_dim, cfg.dropout)
            refit, _, _ = train_model(refit, make_loader(ds_tr, shuffle=True, seed=seed,
                                                          batch_size=cfg.batch_size, num_workers=cfg.num_workers),
                                      None, seed, f"{name} refit", cfg, fixed_epochs=refit_ep)
            torch.save(refit.state_dict(),
                       os.path.join(folder, f"model_seed{seed}_refit.pt"))
            oof[name] = oof_v
            test_preds[name] = predict(refit, te_ld)
            per_seed[name].append({"seed": seed, "refit_epoch": refit_ep,
                                   **compute_metrics(yte, test_preds[name])})
            del refit
            if DEVICE == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        a, ens_m, sweep = alpha_sweep(
            oof["graph_only"], oof["esm_only"], ytr,
            test_preds["graph_only"], test_preds["esm_only"], yte, cfg.alpha_steps)
        per_seed["ensemble"].append({"seed": seed, "alpha": a, **ens_m})
        sweep.assign(seed=seed).to_csv(
            os.path.join(mdir(cfg.output_dir, "ensemble"), f"alpha_sweep_seed{seed}.csv"), index=False)
        print(f"    ensemble: alpha={a:.2f} (graph {a*100:.0f}% / esm {(1-a)*100:.0f}%)  "
              f"R2={ens_m['R2']:.4f} AUC={ens_m['AUC']:.4f} MCC={ens_m['MCC']:.4f}")

        pd.DataFrame({
            "ID": test_full[id_col].values, "y_true": yte,
            "graph": test_preds["graph_only"], "esm": test_preds["esm_only"],
            "hybrid": test_preds["gated_hybrid"],
            "ensemble": np.clip(a * test_preds["graph_only"]
                                + (1 - a) * test_preds["esm_only"], 0, 1),
        }).to_csv(os.path.join(cfg.output_dir, "predictions",
                               f"test_preds_seed{seed}.csv"), index=False)
        pd.DataFrame({
            "ID": train_full[id_col].values, "y_true": ytr,
            "graph_oof": oof["graph_only"], "esm_oof": oof["esm_only"],
        }).to_csv(os.path.join(cfg.output_dir, "predictions",
                               f"oof_preds_seed{seed}.csv"), index=False)

    # write per-model + final CSVs
    pd.DataFrame(gate_hist).to_csv(
        os.path.join(mdir(cfg.output_dir, "gated_hybrid"), "gate_history.csv"), index=False)
    for name in BASE_MODELS:
        pd.DataFrame(cv_epoch_rows[name]).to_csv(
            os.path.join(mdir(cfg.output_dir, name), "cv_epochs.csv"), index=False)

    stats = {}
    for name, rows in per_seed.items():
        folder = mdir(cfg.output_dir, name)
        stats[name] = write_model_summary(folder, rows)
        zip_dir(folder, cfg.zip_outputs)

    final_rows = []
    for name, (mean, sd) in stats.items():
        r = {"model": name}
        for k in KEYS:
            r[k], r[k + "_sd"] = round(float(mean[k]), 4), round(float(sd[k]), 4)
        final_rows.append(r)
    pd.DataFrame(final_rows).to_csv(
        os.path.join(cfg.output_dir, "final_summary.csv"), index=False)

    zip_dir(os.path.join(cfg.output_dir, "predictions"), cfg.zip_outputs)
    if cfg.zip_outputs:
        shutil.make_archive(cfg.output_dir, "zip", root_dir=cfg.output_dir)

    # print table
    show = ["R2", "Pearson", "RMSE", "ACC", "Precision", "Recall", "F1", "AUC", "MCC"]
    w = 26
    print("\n" + "=" * 118)
    print("BioGraphX-Sol NN - eSOL, ProtSATT protocol - 5-seed mean +/- sd (ddof=1)")
    print("=" * 118)
    print(f"  {'model':<{w}}" + "".join(f"{k:>13}" for k in show))
    print("  " + "-" * (w + 13 * len(show)))
    for r in final_rows:
        cells = ""
        for k in show:
            v, s = r.get(k, np.nan), r.get(k + "_sd", np.nan)
            cells += ("        --    " if v != v else
                      (f"{v:>13.4f}" if s != s else f"{v:>7.4f}+/-{s:.3f}"))
        print(f"  {r['model']:<{w}}{cells}")

    print("\n  Per-seed (ensemble):")
    print(f"  {'seed':<8}{'R2':>9}{'AUC':>9}{'MCC':>9}{'Recall':>9}{'alpha':>9}")
    for row in per_seed["ensemble"]:
        print(f"  {row['seed']:<8}{row['R2']:>9.4f}{row['AUC']:>9.4f}"
              f"{row['MCC']:>9.4f}{row['Recall']:>9.4f}{row['alpha']:>9.2f}")

    if gate_hist:
        g = pd.DataFrame(gate_hist)
        last = g.groupby(["seed", "fold"]).last()
        print(f"\n  Gated-hybrid final gates (mean over folds/seeds): "
              f"physics={last['g_physics'].mean():.3f}  esm={last['g_esm'].mean():.3f}")

    print(f"\nDone. -> {cfg.output_dir}/  (final_summary.csv, "
          f"<model>/, predictions/)" + ("  +  " + cfg.output_dir + ".zip" if cfg.zip_outputs else ""))


def main():
    parser = argparse.ArgumentParser(description="Train graph_only / esm_only / gated_hybrid + ensemble on eSOL.")
    parser.add_argument("--train-csv", default="eSol_train_encoded.csv")
    parser.add_argument("--test-csv", default="eSol_test_encoded.csv")
    parser.add_argument("--esm-dirs", nargs="+", required=True,
                        help="One or more directories of {id}.npz ESM-2 embeddings "
                             "(train and test embeddings may live in the same or separate dirs).")
    parser.add_argument("--output-dir", default="biographx_sol_nn_results")
    parser.add_argument("--seeds", type=int, nargs="+", default=[2024, 2025, 2026, 2027, 2028])
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--epoch-select", choices=["r2", "mse"], default="r2",
                        help="'r2' (ProtSATT) or 'mse'; on a fixed val set these pick the "
                             "same epoch - the real lever is --patience.")
    parser.add_argument("--expected-test-rows", type=int, default=660,
                        help="Sanity check on the test CSV row count. Pass 0 to disable.")
    parser.add_argument("--zip-outputs", action="store_true", default=True)
    parser.add_argument("--no-zip-outputs", dest="zip_outputs", action="store_false")

    # Training hyperparameters
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0,
                        help="ESM is pre-pooled into RAM -> no per-item IO, 0 is fine (default: 0).")
    parser.add_argument("--lr-base", type=float, default=5e-5)
    parser.add_argument("--lr-physics", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=300,
                        help="ProtSATT runs 500 w/o early stop; 300 + patience is close & cheaper.")
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--enc-epochs", type=int, default=25,
                        help="Physics-encouragement loss active for the first N epochs (gated_hybrid).")
    parser.add_argument("--enc-lambda", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.3)

    args = parser.parse_args()
    args.alpha_steps = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    if args.expected_test_rows == 0:
        args.expected_test_rows = None
    run(args)


if __name__ == "__main__":
    main()
