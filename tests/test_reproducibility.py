"""(4) Reproducible: fixed seeds, recorded versions, same split."""
import numpy as np

from iot_detector import FEATURES, IoTDetector
from iot_detector.data import split_by_file
from iot_detector.demo import make_synthetic


def test_same_seed_same_model(split):
    train, test = split
    # n_jobs=1: exact. (With n_jobs=-1 parallel summation order can change the last digit, ~1e-16.)
    a = IoTDetector(n_estimators=20, n_jobs=1).fit(train[FEATURES], train["y"])
    b = IoTDetector(n_estimators=20, n_jobs=1).fit(train[FEATURES], train["y"])
    np.testing.assert_array_equal(a.predict_proba(test[FEATURES]), b.predict_proba(test[FEATURES]))
    c = IoTDetector(n_estimators=20).fit(train[FEATURES], train["y"])
    np.testing.assert_allclose(a.predict_proba(test[FEATURES]), c.predict_proba(test[FEATURES]), atol=1e-12)


def test_different_seed_changes_forest(split):
    train, test = split
    a = IoTDetector(n_estimators=20, seed=1).fit(train[FEATURES], train["y"])
    b = IoTDetector(n_estimators=20, seed=2).fit(train[FEATURES], train["y"])
    assert not np.array_equal(a.predict_proba(test[FEATURES]), b.predict_proba(test[FEATURES]))


def test_split_is_by_file_and_repeatable(synth):
    tr1, te1 = split_by_file(synth)
    tr2, te2 = split_by_file(synth)
    assert set(tr1["source_file"]).isdisjoint(set(te1["source_file"]))
    assert tr1.equals(tr2) and te1.equals(te2)
    assert len(tr1) + len(te1) == len(synth)


def test_synthetic_data_repeatable():
    assert make_synthetic(n=200, seed=3).equals(make_synthetic(n=200, seed=3))


def test_versions_recorded(split):
    train, _ = split
    meta = IoTDetector(n_estimators=5).fit(train[FEATURES], train["y"]).meta_
    for key in ("python", "numpy", "pandas", "scikit-learn", "iot_detector"):
        assert key in meta["versions"]
    assert meta["seed"] == 42
