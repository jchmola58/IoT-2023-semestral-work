# Data

**This repo contains no CICIoT2023 data.** The files are large and the dataset
has its own licence, so they must be downloaded by you.

## Where to get it

- Dataset: **CICIoT2023**, Canadian Institute for Cybersecurity (University of New Brunswick).
- Page: https://www.unb.ca/cic/datasets/iotdataset-2023.html
- Download the **CSV** files (`Merged01.csv` ... `Merged63.csv`, 63 files).
- Please read the licence / terms on that page before you use or share the data.
- Paper: Neto et al., "CICIoT2023: A real-time dataset and benchmark for
  large-scale attacks in IoT environment", Sensors 2023.

## What I did with it (Part II)

1. Put the 63 CSV files in one folder, for example `data/raw/` (git ignores it).
2. Keep a random 5% of each file (`random_state=42`) and add a column `source_file`.
3. Add `y` = 1 if `Label` is not `BENIGN`, else 0.
4. Save as `data/ciciot_sample_5pct.parquet` (about 2.25 million rows).

You can do steps 1-4 with the package:

```python
from iot_detector.data import load_csv_folder
df = load_csv_folder("data/raw", frac=0.05, seed=42)
df.to_parquet("data/ciciot_sample_5pct.parquet")   # needs: pip install pyarrow
```

## Columns used

39 numeric columns (see `FEATURES` in `src/iot_detector/features.py`).
`Label`, `y` and `source_file` are **not** model inputs.

## No synthetic data confusion

`iot_detector.demo.make_synthetic` makes small **invented** data for tests and
the demo notebook. It is not real traffic.
