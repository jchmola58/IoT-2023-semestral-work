"""Input checks. Bad inputs give a warning or a refusal, never a silent guess."""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import FEATURES, TARGET_LIKE_COLUMNS, VALID_RANGES


@dataclass
class ValidationReport:
    status: pd.Series                      # "ok" | "repaired" | "refused" per row
    reason: pd.Series                      # short text per row
    messages: list = field(default_factory=list)

    @property
    def usable(self) -> pd.Series:
        return self.status != "refused"


def validate_input(X, known_protocols=None, strict=False, max_bad_fraction=0.2):
    """Check a table of traffic records.

    Rules
    - not a table, empty, or a feature column missing -> ValueError (refuse all)
    - label-like columns (Label, y, source_file)      -> warn and drop them
    - text / inf / NaN / out-of-range values          -> warn, set to NaN
      (the model later fills NaN with the training median), row = "repaired"
    - more than `max_bad_fraction` of a row's features bad -> row "refused"
    - Protocol Type never seen in training            -> row "refused"
    - strict=True: any problem raises ValueError instead of warning

    Returns (clean_features_dataframe, ValidationReport).
    """
    if isinstance(X, dict):
        X = pd.DataFrame(X)
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame (one row per record).")
    if len(X) == 0:
        raise ValueError("Input has no rows.")

    msgs = []

    def problem(text):
        if strict:
            raise ValueError(text)
        msgs.append(text)
        warnings.warn(text, UserWarning, stacklevel=3)

    missing = [c for c in FEATURES if c not in X.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    leaked = [c for c in TARGET_LIKE_COLUMNS if c in X.columns]
    if leaked:
        problem(f"Ignoring columns that are not valid inputs: {leaked}")

    clean = X[FEATURES].copy()
    bad = pd.DataFrame(False, index=clean.index, columns=FEATURES)

    for col in FEATURES:
        raw = clean[col]
        num = pd.to_numeric(raw, errors="coerce").astype("float64")
        not_number = num.isna() & raw.notna()
        if not_number.any():
            problem(f"{int(not_number.sum())} non-numeric value(s) in '{col}' set to missing.")
        is_inf = np.isinf(num)
        if is_inf.any():
            problem(f"{int(is_inf.sum())} infinite value(s) in '{col}' set to missing.")
            num[is_inf] = np.nan
        was_missing = raw.isna()
        if was_missing.any():
            problem(f"{int(was_missing.sum())} missing value(s) in '{col}'.")
        lo, hi = VALID_RANGES[col]
        out = pd.Series(False, index=num.index)
        if lo is not None:
            out |= num < lo
        if hi is not None:
            out |= num > hi
        if out.any():
            problem(f"{int(out.sum())} out-of-range value(s) in '{col}' (allowed {lo}..{hi}) set to missing.")
            num[out] = np.nan
        bad[col] = num.isna()
        clean[col] = num

    frac_bad = bad.mean(axis=1)
    status = pd.Series("ok", index=clean.index, dtype=object)
    reason = pd.Series("", index=clean.index, dtype=object)
    status[frac_bad > 0] = "repaired"
    reason[frac_bad > 0] = "some values missing or invalid; filled with training median"
    too_bad = frac_bad > max_bad_fraction
    status[too_bad] = "refused"
    reason[too_bad] = f"more than {max_bad_fraction:.0%} of features missing or invalid"

    if known_protocols is not None:
        proto = clean["Protocol Type"]
        unseen = proto.notna() & ~proto.isin(list(known_protocols))
        if unseen.any():
            problem(f"{int(unseen.sum())} row(s) with a Protocol Type not seen in training; refused.")
            status[unseen] = "refused"
            reason[unseen] = "unseen Protocol Type"

    return clean, ValidationReport(status=status, reason=reason, messages=msgs)
