"""
Inference layer for the specialist risk models.

For each panel the router (src/ml/router.py) says is eligible:
  1. build the normalized-distance feature vector (src/ml/features.py)
  2. load the trained XGBoost booster for that panel
  3. predict a risk probability
  4. explain the prediction with SHAP (which features drove it, and which
     direction)

This is explicitly a *secondary risk-flagging signal*, not a diagnosis —
callers (src/api.py) are expected to surface it alongside, never instead
of, the existing rule-based severity from reference_ranges.py.
"""
import json
import os
from typing import Dict, List

import numpy as np
import xgboost as xgb

from src.ml.schemas import PANELS
from src.ml.router import route_with_coverage
from src.ml.features import build_feature_vector

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")

_boosters: Dict[str, xgb.Booster] = {}
_metas: Dict[str, dict] = {}


def _load_panel(panel_name: str):
    if panel_name in _boosters:
        return _boosters[panel_name], _metas[panel_name]

    model_path = os.path.join(MODELS_DIR, f"{panel_name}.json")
    meta_path = os.path.join(MODELS_DIR, f"{panel_name}_meta.json")
    if not (os.path.exists(model_path) and os.path.exists(meta_path)):
        return None, None

    booster = xgb.Booster()
    booster.load_model(model_path)
    with open(meta_path) as f:
        meta = json.load(f)

    _boosters[panel_name] = booster
    _metas[panel_name] = meta
    return booster, meta


def _shap_top_features(booster: xgb.Booster, feature_vector: np.ndarray,
                        feature_names: List[str], top_k: int = 3) -> List[dict]:
    """Use XGBoost's built-in SHAP contribution output (pred_contribs=True) —
    avoids an extra heavyweight `shap` dependency at inference time while
    giving the same additive feature-attribution values."""
    d = xgb.DMatrix(feature_vector.reshape(1, -1), feature_names=feature_names, missing=np.nan)
    contribs = booster.predict(d, pred_contribs=True)[0]  # last value = bias term
    pairs = list(zip(feature_names, contribs[:-1]))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return [
        {"feature": name, "contribution": float(val), "direction": "increases_risk" if val > 0 else "decreases_risk"}
        for name, val in pairs[:top_k] if abs(val) > 1e-6
    ]


def predict_all_panels(extracted_values: Dict[str, float], gender: str = "male") -> dict:
    """
    Returns, per panel:
      - eligible: bool (did it pass the coverage gate?)
      - covered_features / total_features
      - risk_probability (None if not eligible or model missing)
      - risk_label
      - top_contributing_features (SHAP-style)
    """
    coverage_report = route_with_coverage(extracted_values)
    output = {}

    for panel_name, spec in PANELS.items():
        cov = coverage_report[panel_name]
        result = {
            "panel_label": spec.label,
            "description": spec.description,
            **cov,
            "risk_probability": None,
            "risk_label": None,
            "top_contributing_features": [],
            "note": None,
        }

        if not cov["eligible"]:
            result["note"] = (
                f"Not enough of this panel's tests were found in the report "
                f"({cov['covered_features']}/{cov['total_features']}, need {cov['required']}). "
                f"Falling back to rule-based reference-range analysis only."
            )
            output[panel_name] = result
            continue

        booster, meta = _load_panel(panel_name)
        if booster is None:
            result["note"] = "Model not trained yet — run `python -m src.ml.train`."
            output[panel_name] = result
            continue

        feature_vector = build_feature_vector(extracted_values, spec.features, gender)
        d = xgb.DMatrix(feature_vector.reshape(1, -1), feature_names=spec.features, missing=np.nan)
        prob = float(booster.predict(d)[0])

        result["risk_probability"] = round(prob, 4)
        result["risk_label"] = spec.positive_class_name if prob >= 0.5 else "normal_pattern"
        result["top_contributing_features"] = _shap_top_features(booster, feature_vector, spec.features)
        output[panel_name] = result

    return output
