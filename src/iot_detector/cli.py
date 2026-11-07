"""Command line: iot-detector train | predict | demo."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .data import load_table, split_by_file
from .demo import make_synthetic
from .detector import IoTDetector
from .features import FEATURES
from .model import evaluate


def _train(a):
    df = load_table(a.data)
    train, test = split_by_file(df, test_size=0.2, seed=a.seed)
    det = IoTDetector(kind=a.model, seed=a.seed).fit(train[FEATURES], train["y"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    det.save(a.out)
    print(f"Saved model to {a.out} (train rows: {len(train)})")
    if a.evaluate:
        res = det.predict(test[FEATURES])
        metrics = evaluate(test["y"], res["proba_malicious"].fillna(0.5))
        metrics["versions"] = det.meta_["versions"]
        print(json.dumps(metrics, indent=2))
        Path(a.out).with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))


def _predict(a):
    det = IoTDetector.load(a.model)
    X = pd.read_csv(a.input)
    res = det.predict(X, strict=a.strict)
    res.to_csv(a.out, index=False)
    print(res["status"].value_counts().to_string())
    print(f"Wrote {a.out}")


def _demo(a):
    df = make_synthetic(n=a.n, seed=0)
    train, test = split_by_file(df)
    det = IoTDetector(n_estimators=50).fit(train[FEATURES], train["y"])
    res = det.predict(test[FEATURES])
    print(evaluate(test["y"], res["proba_malicious"].fillna(0.5)))
    print("(synthetic data: numbers say nothing about real traffic)")


def main(argv=None):
    p = argparse.ArgumentParser(prog="iot-detector")
    sub = p.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train", help="train on the CICIoT2023 sample")
    t.add_argument("--data", required=True, help=".parquet or .csv with Label and source_file")
    t.add_argument("--out", default="models/rf.joblib")
    t.add_argument("--model", choices=["rf", "logistic"], default="rf")
    t.add_argument("--seed", type=int, default=42)
    t.add_argument("--evaluate", action="store_true", help="also score the held-out files")
    t.set_defaults(fn=_train)
    q = sub.add_parser("predict", help="score a CSV of records")
    q.add_argument("--model", required=True)
    q.add_argument("--input", required=True)
    q.add_argument("--out", default="predictions.csv")
    q.add_argument("--strict", action="store_true", help="raise on any bad input")
    q.set_defaults(fn=_predict)
    d = sub.add_parser("demo", help="run on synthetic data")
    d.add_argument("--n", type=int, default=3000)
    d.set_defaults(fn=_demo)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
