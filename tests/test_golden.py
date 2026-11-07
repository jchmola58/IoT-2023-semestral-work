"""(1) Same predictions as the Part II model.

Part II code is re-written here in plain scikit-learn, independent of the
package. The package must give the same numbers on the saved cases.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from iot_detector import FEATURES, IoTDetector

GOLDEN_DIR = Path(__file__).parent / "data"


def _inf_to_nan(X):
    X = np.asarray(X, dtype="float64").copy()
    X[~np.isfinite(X)] = np.nan
    return X


def part2_rf(n):
    return Pipeline([("inf2nan", FunctionTransformer(_inf_to_nan)),
                     ("impute", SimpleImputer(strategy="median")),
                     ("model", RandomForestClassifier(n_estimators=n, min_samples_leaf=5,
                                                      max_features=10, random_state=42, n_jobs=-1))])


def part2_logistic():
    return Pipeline([("inf2nan", FunctionTransformer(_inf_to_nan)),
                     ("impute", SimpleImputer(strategy="median")),
                     ("log1p", FunctionTransformer(np.log1p)),
                     ("scale", StandardScaler()),
                     ("model", LogisticRegression(C=0.03, solver="lbfgs", max_iter=2000, tol=1e-6))])


def test_rf_matches_part2_code(split):
    train, test = split
    ref = part2_rf(40).fit(train[FEATURES], train["y"])
    det = IoTDetector(kind="rf", n_estimators=40).fit(train[FEATURES], train["y"])
    cases = test[FEATURES].head(25)
    np.testing.assert_allclose(det.predict_proba(cases),
                               ref.predict_proba(cases)[:, 1], rtol=0, atol=1e-12)


def test_logistic_matches_part2_code(split):
    train, test = split
    ref = part2_logistic().fit(train[FEATURES], train["y"])
    det = IoTDetector(kind="logistic").fit(train[FEATURES], train["y"])
    cases = test[FEATURES].head(25)
    np.testing.assert_allclose(det.predict_proba(cases),
                               ref.predict_proba(cases)[:, 1], rtol=0, atol=1e-9)


def test_saved_model_gives_same_predictions(split, tmp_path):
    train, test = split
    det = IoTDetector(n_estimators=20).fit(train[FEATURES], train["y"])
    det.save(tmp_path / "m.joblib")
    again = IoTDetector.load(tmp_path / "m.joblib")
    cases = test[FEATURES].head(25)
    np.testing.assert_array_equal(det.predict_proba(cases), again.predict_proba(cases))


@pytest.mark.skipif(not (GOLDEN_DIR / "golden_part2.csv").exists(),
                    reason="real golden cases not created yet (scripts/make_golden_cases.py)")
def test_real_golden_cases_match_part2_notebook():
    """Rows of the real data + probabilities from the Part II notebook model."""
    model_path = Path(__file__).parents[1] / "models" / "rf.joblib"
    if not model_path.exists():
        pytest.skip("models/rf.joblib missing: run `iot-detector train` on the real data first")
    gold = pd.read_csv(GOLDEN_DIR / "golden_part2.csv")
    det = IoTDetector.load(model_path)
    got = det.predict_proba(gold[FEATURES])
    np.testing.assert_allclose(got, gold["expected_proba_malicious"], atol=1e-9)
