"""Pipelines (same steps as the Part II notebook) and evaluation helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, log_loss,
                             precision_recall_fscore_support)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

SEED = 42


def inf_to_nan(X):
    """Turn +-inf into NaN so the imputer can fill them."""
    X = np.array(X, dtype="float64")
    X[~np.isfinite(X)] = np.nan
    return X


def log1p_clip(X):
    """log1p for non-negative data (negative values are clipped to 0)."""
    return np.log1p(np.clip(X, 0, None))


def make_rf(n_estimators=300, min_samples_leaf=5, max_features=10, seed=SEED, n_jobs=-1):
    """Final model of Part II (random forest)."""
    return Pipeline([
        ("inf2nan", FunctionTransformer(inf_to_nan)),
        ("impute", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(
            n_estimators=n_estimators, min_samples_leaf=min_samples_leaf,
            max_features=max_features, random_state=seed, n_jobs=n_jobs)),
    ])


def make_logistic(C=0.03, seed=SEED):
    """Baseline of Part II (L2 logistic regression)."""
    return Pipeline([
        ("inf2nan", FunctionTransformer(inf_to_nan)),
        ("impute", SimpleImputer(strategy="median")),
        ("log1p", FunctionTransformer(log1p_clip)),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=C, solver="lbfgs", max_iter=2000, tol=1e-6,
                                     random_state=seed)),
    ])


def make_pipeline(kind="rf", **kw):
    if kind == "rf":
        return make_rf(**kw)
    if kind == "logistic":
        return make_logistic(**kw)
    raise ValueError("kind must be 'rf' or 'logistic'")


def evaluate(y_true, p_malicious, threshold=0.5):
    """Metrics used in the report. BENIGN is the rare class (about 2%)."""
    y_true = np.asarray(y_true)
    p = np.clip(np.asarray(p_malicious, dtype=float), 1e-15, 1 - 1e-15)
    pred = (p >= threshold).astype(int)
    prec, rec, f1, _ = precision_recall_fscore_support(
        1 - y_true, 1 - pred, average="binary", zero_division=0)
    return {
        "n": int(len(y_true)),
        "log_loss": float(log_loss(y_true, p, labels=[0, 1])),
        "benign_pr_auc": float(average_precision_score(1 - y_true, 1 - p)),
        "benign_precision": float(prec),
        "benign_recall": float(rec),
        "benign_f1": float(f1),
        "threshold": threshold,
    }


def calibration_table(y_true, p_malicious, bins=10):
    """Mean predicted vs observed rate per probability bin (honest check)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(p_malicious, dtype=float)})
    df["bin"] = pd.cut(df["p"], np.linspace(0, 1, bins + 1), include_lowest=True)
    out = df.groupby("bin", observed=True).agg(
        n=("y", "size"), mean_predicted=("p", "mean"), observed_rate=("y", "mean"))
    return out.reset_index()
