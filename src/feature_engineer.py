"""
AutoML Debugger — Feature Engineering Suggestions  (Stage 2 v3.0)
==================================================================
Analyses the dataset and suggests concrete feature transformations:
  - Log transform for skewed numeric features
  - Binning for continuous features with non-linear relationship
  - Interaction features between top correlated pairs
  - Target encoding suggestion for high-cardinality categoricals
  - Datetime decomposition for date columns
  - Polynomial features for small feature sets
  - Drop suggestions for low-importance features
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


def suggest_feature_engineering(
    df: pd.DataFrame,
    target_column: str,
    feature_importance: dict[str, float],
    profile: dict,
    task_type: str,
) -> list[dict[str, str]]:
    """
    Returns list of suggestions:
      [{type, column, suggestion, reason, priority, code_snippet}]
    priority: 'HIGH' | 'MEDIUM' | 'LOW'
    """
    suggestions: list[dict[str, str]] = []
    X = df.drop(columns=[target_column])
    y = df[target_column]

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = X.select_dtypes(exclude=[np.number]).columns.tolist()

    fi_keys = list(feature_importance.keys())
    low_importance = [k for k, v in feature_importance.items() if v < 0.01]

    # ── 1. LOG TRANSFORM for skewed features ─────────────────────
    skewness = profile.get("skewness", {})
    for col in numeric_cols:
        sk = skewness.get(col, 0)
        if abs(sk) > 1.5 and col in X.columns:
            # Only suggest if all values are positive
            if X[col].min() > 0:
                suggestions.append({
                    "type":     "Log Transform",
                    "column":   col,
                    "suggestion": f"Apply log1p transform to '{col}'",
                    "reason":   f"Skewness = {sk:.2f} (|skew| > 1.5). Highly skewed features hurt linear models and can slow tree convergence. Log transform compresses the tail.",
                    "priority": "HIGH" if abs(sk) > 3 else "MEDIUM",
                    "code_snippet": f"df['{col}_log'] = np.log1p(df['{col}'])",
                })

    # ── 2. BINNING for continuous features ───────────────────────
    for col in numeric_cols[:8]:  # top 8 only
        if col in fi_keys[:5]:  # only high-importance features
            n_unique = X[col].nunique()
            if n_unique > 50:
                suggestions.append({
                    "type":     "Binning",
                    "column":   col,
                    "suggestion": f"Create quantile bins for '{col}'",
                    "reason":   f"'{col}' is a top feature with {n_unique} unique values. Binning into 5–10 quantile buckets can help models capture non-linear patterns.",
                    "priority": "MEDIUM",
                    "code_snippet": f"df['{col}_bin'] = pd.qcut(df['{col}'], q=5, labels=False, duplicates='drop')",
                })

    # ── 3. INTERACTION FEATURES between top-2 features ───────────
    if len(fi_keys) >= 2 and len(numeric_cols) >= 2:
        top2 = [f for f in fi_keys[:4] if f in numeric_cols][:2]
        if len(top2) == 2:
            a, b = top2[0], top2[1]
            suggestions.append({
                "type":     "Interaction Feature",
                "column":   f"{a} × {b}",
                "suggestion": f"Create interaction: '{a}' × '{b}'",
                "reason":   f"Both '{a}' and '{b}' are top features. Their product can capture multiplicative relationships the model cannot express independently.",
                "priority": "MEDIUM",
                "code_snippet": f"df['{a}_x_{b}'] = df['{a}'] * df['{b}']",
            })

    # ── 4. TARGET ENCODING for high-cardinality categoricals ─────
    for col in cat_cols:
        n_unique = X[col].nunique()
        if 10 < n_unique <= 50:
            suggestions.append({
                "type":     "Target Encoding",
                "column":   col,
                "suggestion": f"Target-encode '{col}' instead of one-hot encoding",
                "reason":   f"'{col}' has {n_unique} categories. One-hot encoding creates {n_unique} sparse columns. Target encoding replaces each category with its mean target value — fewer features, better signal.",
                "priority": "HIGH" if n_unique > 30 else "MEDIUM",
                "code_snippet": (
                    f"target_means = df.groupby('{col}')['{target_column}'].mean()\n"
                    f"df['{col}_encoded'] = df['{col}'].map(target_means)"
                ),
            })

    # ── 5. DATETIME DECOMPOSITION ─────────────────────────────────
    for col in cat_cols:
        try:
            parsed = pd.to_datetime(X[col], format="mixed", dayfirst=False)
            if parsed.notna().mean() > 0.9:
                suggestions.append({
                    "type":     "Datetime Decomposition",
                    "column":   col,
                    "suggestion": f"Extract year, month, day, dayofweek from '{col}'",
                    "reason":   f"'{col}' is a date column. Raw date strings are useless to ML models, but extracted components (month, weekday, etc.) capture seasonality and temporal patterns.",
                    "priority": "HIGH",
                    "code_snippet": (
                        f"df['{col}'] = pd.to_datetime(df['{col}'])\n"
                        f"df['{col}_year']      = df['{col}'].dt.year\n"
                        f"df['{col}_month']     = df['{col}'].dt.month\n"
                        f"df['{col}_dayofweek'] = df['{col}'].dt.dayofweek\n"
                        f"df['{col}_quarter']   = df['{col}'].dt.quarter"
                    ),
                })
        except Exception:
            pass

    # ── 6. DROP LOW-IMPORTANCE FEATURES ──────────────────────────
    if low_importance:
        suggestions.append({
            "type":     "Drop Low-Importance Features",
            "column":   ", ".join(low_importance[:5]) + ("..." if len(low_importance) > 5 else ""),
            "suggestion": f"Drop {len(low_importance)} feature(s) with importance < 0.01",
            "reason":   "These features contribute almost nothing to predictions but add noise, increase training time, and risk overfitting. Dropping them often improves CV scores.",
            "priority": "LOW",
            "code_snippet": f"cols_to_drop = {low_importance[:5]}\ndf = df.drop(columns=cols_to_drop)",
        })

    # ── 7. POLYNOMIAL FEATURES for small feature sets ────────────
    if len(numeric_cols) <= 8 and task_type == "regression":
        suggestions.append({
            "type":     "Polynomial Features",
            "column":   "All numeric features",
            "suggestion": "Try degree-2 polynomial features for linear model improvement",
            "reason":   f"With only {len(numeric_cols)} numeric features, adding squared terms and cross-products can significantly improve Ridge regression performance at low computational cost.",
            "priority": "LOW",
            "code_snippet": (
                "from sklearn.preprocessing import PolynomialFeatures\n"
                "poly = PolynomialFeatures(degree=2, include_bias=False)\n"
                "X_poly = poly.fit_transform(X_numeric)"
            ),
        })

    # Sort: HIGH first, then MEDIUM, then LOW
    order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    suggestions.sort(key=lambda x: order.get(x["priority"], 3))

    return suggestions
