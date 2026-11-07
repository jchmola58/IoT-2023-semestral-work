"""IoTDetector: fit, save, load, predict with input checks."""
from __future__ import annotations

import platform
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn

from .features import FEATURES, LABELS
from .model import SEED, make_pipeline
from .validation import validate_input


def _versions():
    import iot_detector
    return {"python": platform.python_version(), "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__,
            "iot_detector": iot_detector.__version__}


class IoTDetector:
    """Binary detector: 1 = MALICIOUS, 0 = BENIGN.

    Output is a probability. It is NOT an interval; see README, "Uncertainty".
    """

    def __init__(self, kind="rf", seed=SEED, threshold=0.5, **model_kwargs):
        self.kind = kind
        self.seed = seed
        self.threshold = threshold
        self.model_kwargs = model_kwargs
        self.pipeline_ = None
        self.meta_ = {}

    # ---- training -------------------------------------------------
    def fit(self, X, y):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            clean, report = validate_input(X, strict=False)
        train_issues = [str(w.message) for w in caught]
        y = np.asarray(y)
        if len(y) != len(clean):
            raise ValueError("X and y must have the same number of rows.")
        if set(np.unique(y)) - {0, 1}:
            raise ValueError("y must contain only 0 (BENIGN) and 1 (MALICIOUS).")
        if len(np.unique(y)) < 2:
            raise ValueError("y needs both classes.")
        self.pipeline_ = make_pipeline(self.kind, seed=self.seed, **self.model_kwargs)
        self.pipeline_.fit(clean, y)
        self.meta_ = {
            "kind": self.kind, "seed": self.seed, "features": list(FEATURES),
            "model_kwargs": self.model_kwargs,
            "known_protocols": sorted(float(v) for v in clean["Protocol Type"].dropna().unique()),
            "n_train": int(len(y)), "benign_share_train": float(np.mean(y == 0)),
            "versions": _versions(),
            "train_input_issues": train_issues,
        }
        return self

    # ---- prediction -----------------------------------------------
    def _check_fitted(self):
        if self.pipeline_ is None:
            raise RuntimeError("Model is not fitted. Call fit() or IoTDetector.load().")

    def predict(self, X, strict=False, max_bad_fraction=0.2):
        """Return a DataFrame: proba_malicious, prediction, status, reason.

        Refused rows get NaN probability and no label.
        """
        self._check_fitted()
        clean, report = validate_input(
            X, known_protocols=self.meta_["known_protocols"],
            strict=strict, max_bad_fraction=max_bad_fraction)
        out = pd.DataFrame(index=clean.index)
        out["proba_malicious"] = np.nan
        ok = report.usable.to_numpy()
        if ok.any():
            proba = self.pipeline_.predict_proba(clean.loc[ok, FEATURES])[:, 1]
            out.loc[ok, "proba_malicious"] = proba
        out["prediction"] = [
            LABELS[int(p >= self.threshold)] if not np.isnan(p) else None
            for p in out["proba_malicious"]]
        out["status"] = report.status
        out["reason"] = report.reason
        return out

    def predict_proba(self, X, **kw):
        """Probability of MALICIOUS (NaN for refused rows)."""
        return self.predict(X, **kw)["proba_malicious"].to_numpy()

    # ---- save / load ----------------------------------------------
    def save(self, path):
        self._check_fitted()
        joblib.dump({"pipeline": self.pipeline_, "meta": self.meta_,
                     "threshold": self.threshold}, path)

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        saved = data["meta"]["versions"]
        now = _versions()
        for lib in ("scikit-learn", "numpy", "pandas"):
            if saved.get(lib) != now[lib]:
                warnings.warn(
                    f"Model was saved with {lib} {saved.get(lib)}, "
                    f"you have {now[lib]}. Predictions may differ.", UserWarning)
        obj = cls(kind=data["meta"]["kind"], seed=data["meta"]["seed"],
                  threshold=data.get("threshold", 0.5), **data["meta"]["model_kwargs"])
        obj.pipeline_ = data["pipeline"]
        obj.meta_ = data["meta"]
        return obj
