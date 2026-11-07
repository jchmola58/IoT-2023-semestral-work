"""(5) Honest uncertainty, plus the command line."""
import numpy as np
import pandas as pd

from iot_detector import FEATURES, IoTDetector
from iot_detector.cli import main
from iot_detector.model import calibration_table, evaluate


def test_no_interval_columns_are_claimed(split):
    train, test = split
    out = IoTDetector(n_estimators=10).fit(train[FEATURES], train["y"]).predict(test[FEATURES].head(3))
    assert list(out.columns) == ["proba_malicious", "prediction", "status", "reason"]


def test_probabilities_are_calibrated_on_average(split):
    train, test = split
    det = IoTDetector(n_estimators=50).fit(train[FEATURES], train["y"])
    p = det.predict_proba(test[FEATURES])
    tab = calibration_table(test["y"], p)
    assert tab["n"].sum() == len(test)
    # overall average predicted rate must be close to the observed rate
    assert abs(p.mean() - test["y"].mean()) < 0.05


def test_evaluate_keys():
    m = evaluate([0, 1, 1, 1], [0.2, 0.9, 0.8, 0.6])
    assert {"log_loss", "benign_pr_auc", "benign_recall"} <= set(m)
    assert m["benign_recall"] == 1.0


def test_cli_train_and_predict(synth, tmp_path, capsys):
    data = tmp_path / "d.csv"
    synth.to_csv(data, index=False)
    main(["train", "--data", str(data), "--out", str(tmp_path / "m.joblib"), "--evaluate"])
    assert (tmp_path / "m.metrics.json").exists()
    synth[FEATURES].head(5).to_csv(tmp_path / "in.csv", index=False)
    main(["predict", "--model", str(tmp_path / "m.joblib"), "--input", str(tmp_path / "in.csv"),
          "--out", str(tmp_path / "out.csv")])
    out = pd.read_csv(tmp_path / "out.csv")
    assert len(out) == 5 and out["proba_malicious"].between(0, 1).all()
