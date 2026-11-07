"""(2) only known inputs, (3) bad inputs handled."""
import warnings

import numpy as np
import pandas as pd
import pytest

from iot_detector import FEATURES, IoTDetector, validate_input


@pytest.fixture(scope="module")
def det(split):
    train, _ = split
    return IoTDetector(n_estimators=30).fit(train[FEATURES], train["y"])


def test_label_columns_are_not_used(split, det):
    _, test = split
    X = test.head(10)                       # contains Label, y, source_file
    with pytest.warns(UserWarning, match="Ignoring columns"):
        a = det.predict_proba(X)
    b = det.predict_proba(X[FEATURES])
    np.testing.assert_array_equal(a, b)     # leaked columns change nothing


def test_features_have_no_target_names():
    assert not {"Label", "y", "source_file", "label"} & set(FEATURES)
    assert len(FEATURES) == 39


def test_model_trained_only_on_features(det):
    assert det.pipeline_.n_features_in_ == 39
    assert list(det.meta_["features"]) == FEATURES


def test_missing_column_is_refused(split, det):
    _, test = split
    with pytest.raises(ValueError, match="Missing required columns"):
        det.predict(test[FEATURES].drop(columns=["Rate"]).head(3))


def test_not_a_table_and_empty_are_refused(det, split):
    with pytest.raises(TypeError):
        det.predict([1, 2, 3])
    with pytest.raises(ValueError, match="no rows"):
        det.predict(split[1][FEATURES].head(0))


def test_nan_is_warned_and_filled(split, det):
    X = split[1][FEATURES].head(5).copy()
    X.loc[X.index[0], "Std"] = np.nan
    with pytest.warns(UserWarning, match="missing value"):
        out = det.predict(X)
    assert out["status"].iloc[0] == "repaired"
    assert out["proba_malicious"].notna().all()


def test_inf_is_warned(split, det):
    X = split[1][FEATURES].head(5).copy()
    X.loc[X.index[1], "Rate"] = np.inf
    with pytest.warns(UserWarning, match="infinite"):
        out = det.predict(X)
    assert out["status"].iloc[1] == "repaired"


def test_out_of_range_is_warned(split, det):
    X = split[1][FEATURES].head(5).copy()
    X.loc[X.index[2], "Header_Length"] = -5
    X.loc[X.index[3], "SSH"] = 7
    with pytest.warns(UserWarning, match="out-of-range"):
        out = det.predict(X)
    assert out["status"].iloc[2] == "repaired"
    assert out["status"].iloc[3] == "repaired"


def test_text_value_is_warned(split, det):
    X = split[1][FEATURES].head(3).copy().astype(object)
    X.loc[X.index[0], "Rate"] = "abc"
    with pytest.warns(UserWarning, match="non-numeric"):
        out = det.predict(X)
    assert out["status"].iloc[0] == "repaired"


def test_unseen_category_is_refused(split, det):
    X = split[1][FEATURES].head(4).copy()
    X.loc[X.index[0], "Protocol Type"] = 99
    with pytest.warns(UserWarning, match="not seen in training"):
        out = det.predict(X)
    assert out["status"].iloc[0] == "refused"
    assert np.isnan(out["proba_malicious"].iloc[0])
    assert pd.isna(out["prediction"].iloc[0])
    assert out["status"].iloc[1:].eq("ok").all()


def test_mostly_bad_row_is_refused(split, det):
    X = split[1][FEATURES].head(3).copy()
    X.iloc[0, 4:30] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = det.predict(X)
    assert out["status"].iloc[0] == "refused"
    assert np.isnan(out["proba_malicious"].iloc[0])


def test_strict_mode_raises(split, det):
    X = split[1][FEATURES].head(3).copy()
    X.loc[X.index[0], "Rate"] = np.inf
    with pytest.raises(ValueError):
        det.predict(X, strict=True)


def test_clean_input_gives_no_warning(split, det):
    X = split[1][FEATURES].head(3).copy()
    X = X.replace([np.inf], 1.0).fillna(1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = det.predict(X)
    assert out["status"].eq("ok").all()


def test_validate_input_directly(split):
    clean, rep = validate_input(split[1][FEATURES].head(3).replace(np.inf, 1.0).fillna(1))
    assert list(clean.columns) == FEATURES and rep.usable.all()
