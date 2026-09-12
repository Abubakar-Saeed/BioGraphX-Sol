#!/usr/bin/env python3
"""
BioGraphX-Sol feature encoding - CLI entrypoint.

Converts a CSV of protein sequences into 200-dimensional, per-protein
biophysical feature vectors for solubility prediction.

Input CSV format
-----------------
    gene,sequence,solubility
    P0A6F5,MKTAYIAKQR...,0.82

`solubility` is optional. If present it is carried through as a `Label`
column in the output. The `gene` column becomes the output `ID`.

Usage
-----
    python run.py --input-file eSol_train.csv \\
                   --output-file eSol_train_encoded.csv \\
                   --n-jobs 8
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from biographx_sol import run_solubility_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Encode protein sequences into 200 per-protein BioGraphX-Sol features."
    )
    parser.add_argument("--input-file", required=True,
                        help="CSV with gene, sequence, and optional solubility columns.")
    parser.add_argument("--output-file", required=True,
                        help="Destination CSV for per-protein features.")
    parser.add_argument("--chunk-size", type=int, default=100,
                        help="Number of sequences grouped per parallel batch (default: 100).")
    parser.add_argument("--n-jobs", type=int, default=4,
                        help="Parallel worker processes for encoding (default: 4).")
    args = parser.parse_args()

    run_solubility_pipeline(
        args.input_file,
        args.output_file,
        chunk_size=args.chunk_size,
        n_jobs=args.n_jobs,
    )


if __name__ == "__main__":
    main()
