"""Part II random forest rebuilt in plain scikit-learn (no iot_detector code).
Writes its probabilities into tests/data/golden_part2.csv."""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

def inf_to_nan(X):
    X = np.asarray(X, dtype="float64").copy()
    X[~np.isfinite(X)] = np.nan
    return X

df = pd.read_parquet("data/ciciot_sample_5pct.parquet")
if "y" not in df.columns:
    df["y"] = (df["Label"] != "BENIGN").astype(int)
feats = [c for c in df.columns if c not in ["Label", "y", "source_file"]]

tr, _ = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
             .split(df, groups=df["source_file"]))
train = df.iloc[tr]

rf = Pipeline([
    ("inf2nan", FunctionTransformer(inf_to_nan)),
    ("impute", SimpleImputer(strategy="median")),
    ("model", RandomForestClassifier(n_estimators=300, min_samples_leaf=5,
                                     max_features=10, random_state=42, n_jobs=-1)),
]).fit(train[feats], train["y"])

path = "tests/data/golden_part2.csv"
g = pd.read_csv(path)
g["expected_proba_malicious"] = rf.predict_proba(g[feats])[:, 1]
g.to_csv(path, index=False, float_format="%.17g")
print(g[["y", "expected_proba_malicious"]])