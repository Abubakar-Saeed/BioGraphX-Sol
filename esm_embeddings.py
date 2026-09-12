#!/usr/bin/env python3
"""
BioGraphX-Sol - ESM-2 embedding generation.

Extracts per-residue ESM-2 embeddings for every sequence in a CSV file and
saves them as one compressed .npz file per protein (`{ID}.npz`, key
`embedding`, shape [seq_len, hidden_dim], dtype float16). These embeddings
are mean-pooled and consumed by the ESM-only and Gated Hybrid streams in
`training/`, and by `inference.py`.

This script is identical across the BioGraphX-* repos (BioGraphX-IDP,
BioGraphX-Sol, ...) - ESM-2 embedding generation from a CSV of sequences is
the same job regardless of what a downstream model does with the result.

Usage
-----
    python esm_embeddings.py --csv-path eSol_train_for_esm.csv \\
                              --output-dir esm_embeddings/eSol_train \\
                              --model-name facebook/esm2_t36_3B_UR50D \\
                              --batch-size 8

Large datasets can be processed in parts (e.g. across multiple sessions/GPUs)
with --part / --total-parts.

Input CSV
---------
Needs `ID` and `Sequence` columns. BioGraphX-Sol's own raw data uses `gene`
and `sequence` - rename them first, e.g.:

    df.rename(columns={"gene": "ID", "sequence": "Sequence"}).to_csv(...)

so the `{ID}.npz` filenames line up with the `ID` column written by
`BioGraphX-Sol-Encoding/run.py`.
"""

import argparse
import gc
import os
from functools import partial

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


class ProteinDataset(Dataset):
    def __init__(self, df):
        self.entry_ids = df['ID'].values
        self.sequences = df['Sequence'].values

    def __len__(self):
        return len(self.entry_ids)

    def __getitem__(self, idx):
        return self.entry_ids[idx], self.sequences[idx]


# Per-process tokenizer + max_length, used by the module-level collate_fn below.
# A DataLoader with num_workers>0 pickles collate_fn by reference (module.name),
# not by value, so collate_fn cannot be a closure over a local `tokenizer` -
# that pattern crashes with "Can't pickle local object" the moment a worker
# process is spawned (the default on Windows/macOS, and optionally on Linux).
# Instead each process (main or worker) fills in its own copy of this holder.
_tokenizer_state = {"tokenizer": None, "max_length": 1024}


def _init_tokenizer_state(model_name: str, max_length: int):
    _tokenizer_state["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
    _tokenizer_state["max_length"] = max_length


def _worker_init_fn(_worker_id, model_name: str, max_length: int):
    _init_tokenizer_state(model_name, max_length)


def collate_fn(batch):
    tokenizer = _tokenizer_state["tokenizer"]
    ids, seqs = zip(*batch)
    inputs = tokenizer(
        list(seqs),
        return_tensors="pt",
        truncation=True,
        max_length=_tokenizer_state["max_length"],
        padding=True,
    )
    return ids, inputs


def extract_esm_embeddings(
    csv_path: str,
    output_dir: str,
    model_name: str = "facebook/esm2_t36_3B_UR50D",
    part: int = 1,
    total_parts: int = 1,
    batch_size: int = 8,
    num_workers: int = 2,
    max_length: int = 1024,
):
    # 1. Setup output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving embeddings to: {output_dir}")

    # 2. Load data & optionally split into parts (for staged/multi-session runs)
    # Split on the row-index array, not the DataFrame itself: np.array_split on
    # a DataFrame silently returns plain ndarrays on current numpy/pandas,
    # which breaks the column-name access (df_part['ID']) below.
    df = pd.read_csv(csv_path)
    total_seqs = len(df)
    index_chunks = np.array_split(np.arange(total_seqs), total_parts)
    df_part = df.iloc[index_chunks[part - 1]].reset_index(drop=True)

    print(f"Total dataset: {total_seqs}")
    print(f"Running part {part}/{total_parts}")
    print(f"Processing {len(df_part)} sequences in this run...")

    # 3. Model setup (multi-GPU aware)
    print(f"Loading {model_name}...")
    _init_tokenizer_state(model_name, max_length)  # tokenizer for the main process itself
    model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float16)

    if torch.cuda.device_count() > 1:
        print(f"{torch.cuda.device_count()} GPUs detected - activating DataParallel.")
        model = torch.nn.DataParallel(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    # 4. Dataset & loader
    dataset = ProteinDataset(df_part)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        worker_init_fn=(
            partial(_worker_init_fn, model_name=model_name, max_length=max_length)
            if num_workers > 0 else None
        ),
        pin_memory=True,
    )

    # 5. Extraction loop
    print("Starting extraction...")
    with torch.no_grad():
        for batch_ids, inputs in tqdm(loader, desc=f"Part {part}"):
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            embeddings = outputs.last_hidden_state.detach().cpu()
            mask = inputs['attention_mask'].cpu()

            for i, entry_id in enumerate(batch_ids):
                save_path = os.path.join(output_dir, f"{entry_id}.npz")
                if os.path.exists(save_path):
                    continue

                # Slice off padding using the attention mask
                valid_len = mask[i].sum().item()
                emb_numpy = embeddings[i, :valid_len, :].numpy().astype(np.float16)
                np.savez_compressed(save_path, embedding=emb_numpy)

            del input_ids, attention_mask, outputs, embeddings
            gc.collect()

    print(f"Part {part} complete.")


def main():
    parser = argparse.ArgumentParser(description="Extract per-residue ESM-2 embeddings for a CSV of sequences.")
    parser.add_argument("--csv-path", required=True, help="CSV with ID and Sequence columns.")
    parser.add_argument("--output-dir", required=True, help="Directory to write one {ID}.npz file per protein.")
    parser.add_argument("--model-name", default="facebook/esm2_t36_3B_UR50D",
                        help="HuggingFace ESM-2 checkpoint (default: facebook/esm2_t36_3B_UR50D).")
    parser.add_argument("--part", type=int, default=1, help="Which part to run (1-indexed, default: 1).")
    parser.add_argument("--total-parts", type=int, default=1,
                        help="Split the dataset into this many parts across runs (default: 1).")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8).")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers (default: 2).")
    parser.add_argument("--max-length", type=int, default=1024, help="Tokenizer truncation length (default: 1024).")
    args = parser.parse_args()

    extract_esm_embeddings(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        model_name=args.model_name,
        part=args.part,
        total_parts=args.total_parts,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_length=args.max_length,
    )


if __name__ == "__main__":
    main()
