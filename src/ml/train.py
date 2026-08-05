"""
Train the 4 specialist panel-risk models.

Model choice: XGBoost (gradient boosted trees).
  Why not logistic regression: LR needs upfront imputation (mean/median +
  missing-indicator columns) before it can even see the data.
  Why XGBoost: at every split, the tree learns a default direction for
  missing values based on what minimizes training loss — missingness is
  handled natively, which is exactly the constraint this project has
  (every blood report has a different subset of tests present).

Run:
    python -m src.ml.train

Outputs:
    src/ml/models/<panel>.json         trained booster
    src/ml/models/<panel>_meta.json    feature order + metrics
    eval/results/ml_model_metrics.json combined report for all 4 panels
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                              recall_score, roc_auc_score, confusion_matrix)
from sklearn.model_selection import train_test_split
import xgboost as xgb

from src.ml.schemas import PANELS
from src.ml.data_synthesis import GENERATORS
from src.ml.features import transform_dataframe_to_features

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "eval", "results", "ml_model_metrics.json")


def train_one_panel(panel_name: str) -> dict:
    spec = PANELS[panel_name]
    df = GENERATORS[panel_name](n=6000)

    # IMPORTANT: train on the exact same normalized-distance feature space
    # used at inference time (src/ml/predict.py -> build_feature_vector).
    # Training on raw values while predicting on normalized distances would
    # silently produce a model that scores nonsense at inference time.
    X = transform_dataframe_to_features(df, spec.features)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=spec.features, missing=np.nan)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=spec.features, missing=np.nan)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "max_depth": 4,
        "eta": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "seed": 42,
    }
    booster = xgb.train(params, dtrain, num_boost_round=150,
                         evals=[(dtest, "test")], verbose_eval=False)

    y_prob = booster.predict(dtest)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        "panel": panel_name,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_test": float(y_test.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": {
            k: float(v) for k, v in booster.get_score(importance_type="gain").items()
        },
    }

    os.makedirs(MODELS_DIR, exist_ok=True)
    booster.save_model(os.path.join(MODELS_DIR, f"{panel_name}.json"))
    with open(os.path.join(MODELS_DIR, f"{panel_name}_meta.json"), "w") as f:
        json.dump({"feature_order": spec.features, "metrics": metrics}, f, indent=2)

    return metrics


def main():
    all_metrics = {}
    for panel_name in PANELS:
        print(f"Training panel: {panel_name} ...")
        m = train_one_panel(panel_name)
        all_metrics[panel_name] = m
        print(f"  AUC={m['roc_auc']:.3f}  F1={m['f1']:.3f}  Precision={m['precision']:.3f}  Recall={m['recall']:.3f}")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved combined metrics to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
