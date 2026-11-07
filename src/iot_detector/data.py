"""Data loading and the file-grouped split used in Part II.

The CICIoT2023 data is NOT part of this repo. See data/README.md.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

SEED = 42


def add_target(df):
    """y = 1 for any attack label, 0 for BENIGN."""
    df = df.copy()
    df["y"] = (df["Label"].astype(str).str.upper() != "BENIGN").astype(int)
    return df


def load_csv_folder(folder, frac=0.05, seed=SEED):
    """Read all Merged*.csv files, keep `frac` of each file, add source_file."""
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No .csv files in {folder}. See data/README.md.")
    parts = []
    for f in files:
        part = pd.read_csv(f).sample(frac=frac, random_state=seed)
        part["source_file"] = f.name
        parts.append(part)
    return add_target(pd.concat(parts, ignore_index=True))


def load_table(path):
    """Load the cached sample (.parquet) or a .csv with Label and source_file."""
    path = Path(path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    return df if "y" in df.columns else add_target(df)


def split_by_file(df, test_size=0.2, seed=SEED):
    """Whole files go to train OR test (no file in both)."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(df, groups=df["source_file"]))
    return df.iloc[tr].reset_index(drop=True), df.iloc[te].reset_index(drop=True)
