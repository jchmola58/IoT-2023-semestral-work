"""Small SYNTHETIC data with the same columns as CICIoT2023.

For tests and the demo notebook only. It is invented data, not real traffic.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import BINARY_FEATURES, FEATURES


def make_synthetic(n=3000, n_files=10, benign_share=0.15, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) > benign_share).astype(int)
    df = pd.DataFrame(index=range(n), columns=FEATURES, dtype="float64")
    df["Header_Length"] = np.where(y == 1, rng.gamma(2, 20000, n), rng.gamma(2, 30, n))
    df["Protocol Type"] = rng.choice([1, 6, 17], n, p=[0.2, 0.5, 0.3]).astype(float)
    df["Time_To_Live"] = rng.choice([64, 128, 255], n).astype(float)
    df["Rate"] = np.where(y == 1, rng.gamma(2, 3000, n), rng.gamma(2, 3, n))
    for c in BINARY_FEATURES:
        p = 0.3
        df[c] = (rng.random(n) < np.where(y == 1, p + 0.2, p)).astype(float)
    for c in ["ack_count", "syn_count", "fin_count", "rst_count"]:
        df[c] = rng.gamma(1.5, 0.3 + 0.5 * y, n)
    for c in ["Tot sum", "Min", "Max", "AVG", "Tot size", "IAT", "Number", "Std", "Variance"]:
        df[c] = rng.gamma(2, 50 + 100 * (y if c in ("Number", "IAT") else 0), n)
    df.loc[rng.choice(n, 3, replace=False), "Rate"] = np.inf   # like the real data
    df.loc[rng.choice(n, 3, replace=False), "Std"] = np.nan
    df["Label"] = np.where(y == 1, "DDOS-SYN_FLOOD", "BENIGN")
    df["y"] = y
    df["source_file"] = [f"Merged{i:02d}.csv" for i in rng.integers(1, n_files + 1, n)]
    return df
