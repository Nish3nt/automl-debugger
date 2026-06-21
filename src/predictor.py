"""
AutoML Debugger — Predictor  (Stage 1 v3.0)
=============================================
Handles predictions on new data using the trained best pipeline.
Supports:
  - CSV upload with predictions
  - Single-row form input
  - Confidence intervals for regression
  - Probability scores for classification
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


def predict_from_dataframe(
    df_new: pd.DataFrame,
    best_pipeline,
    label_encoder,
    task_type: str,
    numeric_features: list[str],
    categorical_features: list[str],
    pred_std: float = 0.0,
) -> pd.DataFrame:
    """
    Run predictions on a new DataFrame.
    Returns df_new with added prediction columns.
    """
    result = df_new.copy()

    # Keep only known columns (drop unknown, fill missing with NaN)
    all_features = numeric_features + categorical_features
    for col in all_features:
        if col not in result.columns:
            result[col] = np.nan

    X = result[all_features]
    preds = best_pipeline.predict(X)

    if task_type == "regression":
        if label_encoder is not None:
            preds = label_encoder.inverse_transform(preds.astype(int))
        result["Prediction"] = np.round(preds, 4)
        if pred_std > 0:
            result["Lower Bound (95%)"] = np.round(preds - 1.96 * pred_std, 4)
            result["Upper Bound (95%)"] = np.round(preds + 1.96 * pred_std, 4)

    else:
        if label_encoder is not None:
            decoded = label_encoder.inverse_transform(preds.astype(int))
        else:
            decoded = preds
        result["Prediction"] = decoded

        # Probability scores
        try:
            probas = best_pipeline.predict_proba(X)
            if label_encoder is not None:
                classes = label_encoder.inverse_transform(range(probas.shape[1]))
            else:
                classes = best_pipeline.classes_

            # Add confidence of predicted class
            result["Confidence"] = np.round(probas.max(axis=1), 4)

            # Add individual class probabilities if binary
            if probas.shape[1] == 2:
                result[f"P({classes[0]})"] = np.round(probas[:, 0], 4)
                result[f"P({classes[1]})"] = np.round(probas[:, 1], 4)
        except Exception:
            pass

    return result


def predict_single_row(
    input_dict: dict,
    best_pipeline,
    label_encoder,
    task_type: str,
    numeric_features: list[str],
    categorical_features: list[str],
    pred_std: float = 0.0,
) -> dict[str, Any]:
    """
    Predict for a single row given as a dict.
    Returns prediction + confidence interval / probabilities.
    """
    row = {col: [input_dict.get(col, np.nan)] for col in numeric_features + categorical_features}
    df_row = pd.DataFrame(row)

    result_df = predict_from_dataframe(
        df_row, best_pipeline, label_encoder,
        task_type, numeric_features, categorical_features, pred_std,
    )

    out: dict[str, Any] = {"prediction": result_df["Prediction"].iloc[0]}

    if task_type == "regression" and pred_std > 0:
        out["lower_bound"] = result_df["Lower Bound (95%)"].iloc[0]
        out["upper_bound"] = result_df["Upper Bound (95%)"].iloc[0]
        out["interval"]    = f"{out['lower_bound']:.4f} – {out['upper_bound']:.4f}"
    elif task_type == "classification":
        if "Confidence" in result_df.columns:
            out["confidence"] = result_df["Confidence"].iloc[0]
        proba_cols = [c for c in result_df.columns if c.startswith("P(")]
        out["probabilities"] = {col: result_df[col].iloc[0] for col in proba_cols}

    return out
