# IoT-2023-semestral-work: `iot-detector`

Detect **MALICIOUS vs BENIGN** IoT network traffic records from the
[CICIoT2023](https://www.unb.ca/cic/datasets/iotdataset-2023.html) dataset.
The model is a random forest (the winner of Part II). It reads 39 numeric
traffic features and returns a **probability of MALICIOUS**.

*University project: Model Development Report, Part III (production, option B).*

**For whom:** someone who has records of IoT traffic features (same columns as
CICIoT2023) and wants a first-pass flag for suspicious records, with clear
warnings when the input looks wrong.

**Not for:** deciding which *type* of attack it is; networks very different from
the CICIoT2023 lab; anything where a missed attack has serious cost without a human check.

## What it does

- Trains the Part II model (imputation of missing/infinite values, then random forest).
- Predicts on a table of records. Output columns: `proba_malicious`, `prediction`, `status`, `reason`.
- Checks inputs: missing columns, empty input, text, infinite, missing, out-of-range values and unseen `Protocol Type`.
- Saves and loads models with seed and library versions.

## Install

Python 3.9 or newer.

```bash
git clone https://github.com/jchmola58/IoT-2023-semestral-work.git
cd IoT-2023-semestral-work
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -e ".[dev]"                                # add ,data for parquet support
```

Or install the release `v0.1.0` straight from GitHub:

```bash
pip install "git+https://github.com/jchmola58/IoT-2023-semestral-work.git@v0.1.0"
```

Versions used for the results are recorded in `requirements-lock.txt` and inside each saved model.

## Use

Try it without any data (synthetic, invented records):

```bash
iot-detector demo
```

With the real data (see [data/README.md](data/README.md) to download it):

```bash
iot-detector train --data data/ciciot_sample_5pct.parquet --out models/rf.joblib --evaluate
iot-detector predict --model models/rf.joblib --input examples/sample_input.csv --out predictions.csv
```

In Python:

```python
from iot_detector import IoTDetector, FEATURES
det = IoTDetector.load("models/rf.joblib")
result = det.predict(df)          # df has the 39 FEATURES columns
```

The notebook [notebooks/demo.ipynb](notebooks/demo.ipynb) shows the full flow on synthetic data.

## Bad inputs

| Input problem | What happens |
|---|---|
| Not a table / no rows / a feature column missing | **Refused** (`ValueError`) |
| `Label`, `y`, `source_file` in the table | Warning, columns **ignored** (never used) |
| Missing, infinite, text or out-of-range value | Warning, set to missing, filled with the training median, row `repaired` |
| More than 20% of a row's features bad | Row **refused** (no probability) |
| `Protocol Type` not seen in training | Row **refused** |
| `strict=True` (or `--strict`) | Any problem raises an error |

## Tests

```bash
pytest -v
```

GitHub Actions runs the same tests after every push (`.github/workflows/ci.yml`).
The tests cover:

1. **Same predictions as Part II.** Part II code is rewritten in plain scikit-learn inside the test; the package must match it to 1e-12. A second test compares against saved real cases (`tests/data/golden_part2.csv`, made by `scripts/make_golden_cases.py`). It is skipped until that file and `models/rf.joblib` exist.
2. **Only inputs known at prediction time.** 39 features, no `Label`/`y`/`source_file`; extra label columns change nothing.
3. **Bad inputs** (table above).
4. **Reproducibility.** Fixed seed 42, same file-grouped split, versions recorded. With `n_jobs=-1` the last digit can differ (about 1e-16) because of parallel summation. With `n_jobs=1` results are identical.
5. **Uncertainty.** See below.
6. **CLI** end to end.

All code was written with AI help (Claude). I do not trust it because of that. I trust it because of these tests.

## Uncertainty (honest version)

- The output is a **probability, not an interval**. No confidence or prediction interval is shown.
- A calibration check (`iot_detector.model.calibration_table`) compares predicted and observed rates.
- Calibration and scores come from files of the **same lab**. New environments can be worse.
- BENIGN is rare (about 2%). Look at BENIGN precision and recall, not only accuracy.
- The default threshold is 0.5. Change `IoTDetector(threshold=...)` if the cost of errors needs it.

## Results (Part II, from the report)

Validation fold inside the training files (random forest): log loss 0.0164,
BENIGN precision / recall / F1 = 0.841 / 0.847 / 0.844. Add the final held-out
numbers from `iot-detector train --evaluate` here (`models/rf.metrics.json`).

## Limits and monitoring

See [docs/monitoring.md](docs/monitoring.md). Main risks: new attack types, drift,
and one-lab data.

## Layout

```
src/iot_detector/   package (features, validation, model, detector, data, cli)
tests/              pytest tests
notebooks/demo.ipynb
data/README.md      where to get the data (no data in the repo)
scripts/            make_golden_cases.py
docs/monitoring.md
.github/workflows/  CI
```

## Licence

Code: MIT. The CICIoT2023 data has its own terms and is **not** included.
