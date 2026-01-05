"""
debugger_engine.py

This file contains the core logic of the AutoML Debugger.
It takes data, trains a model, diagnoses issues,
and generates human-readable explanations.
"""

import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def run_debugger_pipeline():
    """
    Main function that runs the complete debugger pipeline.
    Returns final debugger output as a dictionary.
    """

    # -----------------------------
    # LOAD DATA
    # -----------------------------
    dataset = load_breast_cancer()
    X = pd.DataFrame(dataset.data, columns=dataset.feature_names)
    y = pd.Series(dataset.target, name="target")

    # -----------------------------
    # TRAIN-VALIDATION SPLIT
    # -----------------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # -----------------------------
    # FEATURE SCALING
    # -----------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # -----------------------------
    # MODEL TRAINING
    # -----------------------------
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    # -----------------------------
    # PERFORMANCE EVALUATION
    # -----------------------------
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    val_acc = accuracy_score(y_val, model.predict(X_val_scaled))
    acc_gap = train_acc - val_acc

    if train_acc > 0.95 and acc_gap > 0.05:
        diagnosis = "Overfitting detected"
    elif train_acc < 0.75 and val_acc < 0.75:
        diagnosis = "Underfitting detected"
    else:
        diagnosis = "Model learning looks healthy"

    # -----------------------------
    # FEATURE IMPORTANCE
    # -----------------------------
    feature_importance = pd.Series(
        model.coef_[0],
        index=X.columns
    ).sort_values(key=abs, ascending=False)

    # -----------------------------
    # ROOT CAUSE LOGIC
    # -----------------------------
    root_causes = []
    recommendations = []

    if acc_gap > 0.05:
        root_causes.append("Model generalization gap detected")
        recommendations.append("Apply regularization or collect more data")

    if feature_importance.abs().max() > 5 * feature_importance.abs().mean():
        root_causes.append("Suspicious feature dominance")
        recommendations.append("Check for data leakage or dominant features")

    if not root_causes:
        root_causes.append("No major issues detected")
        recommendations.append("Model pipeline looks healthy")

    # -----------------------------
    # GENAI-STYLE EXPLANATION
    # -----------------------------
    explanations = []
    for i, cause in enumerate(root_causes):
        explanations.append(
            f"Issue detected: {cause}. "
            f"Recommended action: {recommendations[i]}. "
            f"This improves model stability and reliability."
        )

    # -----------------------------
    # FINAL OUTPUT
    # -----------------------------
    return {
        "train_accuracy": round(train_acc, 4),
        "validation_accuracy": round(val_acc, 4),
        "diagnosis": diagnosis,
        "root_causes": root_causes,
        "recommendations": recommendations,
        "explanations": explanations
    }
