"""
Shared feature engineering for all specialist risk models.

Every lab value is converted to a "normalized distance from the reference
range midpoint" feature. This does two things:
  1. Puts every test on a comparable scale regardless of unit
     (glucose in mg/dL vs. albumin in g/dL become comparable floats).
  2. Reuses the reference-range logic already defined in reference_ranges.py,
     so the ML layer sits on top of the existing rule engine rather than
     duplicating it.

feature = (value - midpoint) / (half_range)
  -> 0.0  means "dead center of normal range"
  -> 1.0  means "exactly at the boundary of normal"
  -> >1.0 means "outside the normal range", magnitude = how far outside

Missing tests are encoded as NaN and left for the tree-based model to
handle natively (see train.py for why XGBoost is used).
"""
from typing import Dict, List, Optional

import numpy as np

from src.reference_ranges import get_range


def normalized_distance(test_name: str, value: float, gender: str = "male") -> Optional[float]:
    ref = get_range(test_name.lower().strip(), gender)
    if not ref or "min" not in ref or "max" not in ref:
        return None
    low, high = ref["min"], ref["max"]
    if high <= low:
        return None
    midpoint = (low + high) / 2
    half_range = (high - low) / 2
    return (value - midpoint) / half_range


def build_feature_vector(extracted_values: Dict[str, float], feature_names: List[str],
                          gender: str = "male") -> np.ndarray:
    """
    Build a fixed-length feature vector for one panel.
    Missing tests -> np.nan (handled natively by XGBoost's split-direction learning).
    """
    row = []
    for name in feature_names:
        if name in extracted_values:
            dist = normalized_distance(name, float(extracted_values[name]), gender)
            row.append(dist if dist is not None else np.nan)
        else:
            row.append(np.nan)
    return np.array(row, dtype="float32")


def coverage(extracted_values: Dict[str, float], feature_names: List[str]) -> int:
    return sum(1 for name in feature_names if name in extracted_values)


def transform_dataframe_to_features(df, feature_names: List[str], gender_col: str = "gender") -> np.ndarray:
    """
    Convert a DataFrame of raw lab values (one column per test, optional
    per-row `gender` column) into the same normalized-distance feature space
    used at inference time (build_feature_vector). Training and prediction
    MUST go through this same transform or the model sees a different
    feature distribution than it was trained on.
    """
    n = len(df)
    genders = df[gender_col].values if gender_col in df.columns else np.array(["male"] * n)
    out = np.full((n, len(feature_names)), np.nan, dtype="float32")
    for j, name in enumerate(feature_names):
        if name not in df.columns:
            continue
        values = df[name].values
        for i in range(n):
            v = values[i]
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            dist = normalized_distance(name, float(v), str(genders[i]))
            out[i, j] = dist if dist is not None else np.nan
    return out
