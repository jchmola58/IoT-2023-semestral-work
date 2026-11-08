"""Create tests/data/golden_part2.csv from the REAL data and the Part II model.

Run once on your computer (needs the parquet file):

    python scripts/make_golden_cases.py --data data/ciciot_sample_5pct.parquet \
        --model models/rf.joblib

Option A (default): `models/rf.joblib` was made by `iot-detector train` (same seed).
Option B: pass --from-notebook probas.csv with the column `expected_proba_malicious`
          for the same rows, taken from your Part II notebook model (`rf_final`).
Here we pick 10 rows from the held-out files with a fixed seed.
"""
import argparse

import pandas as pd

from iot_detector import FEATURES, IoTDetector
from iot_detector.data import load_table, split_by_file

p = argparse.ArgumentParser()
p.add_argument("--data", required=True)
p.add_argument("--model", default="models/rf.joblib")
p.add_argument("--out", default="tests/data/golden_part2.csv")
p.add_argument("--n", type=int, default=10)
a = p.parse_args()

_, test = split_by_file(load_table(a.data))
rows = pd.concat([test[test["y"] == 0].sample(5, random_state=0),
                  test[test["y"] == 1].sample(5, random_state=0)]
                 )[FEATURES + ["y"]].reset_index(drop=True)
rows["expected_proba_malicious"] = IoTDetector.load(a.model).predict_proba(rows[FEATURES])
rows.to_csv(a.out, index=False, float_format="%.17g")
print(f"Wrote {a.out}. Replace expected_proba_malicious with the Part II notebook "
      "values (rf_final.predict_proba) to compare against the notebook itself.")
