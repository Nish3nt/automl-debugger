"""
AutoML Debugger Engine
======================
Production-grade ML diagnostics pipeline.

Supports:
- Auto-detection of task type (classification vs regression)
- Real LLM-powered analysis via Anthropic Claude
- Rich data quality profiling
- Feature importance via tree-based models
- Outlier, duplicate, and class imbalance detection
"""

from __future__ import annotations

import os
import json
import traceback
from typing import Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score, classification_report
)


# ─────────────────────────────────────────────
# Task auto-detection
# ─────────────────────────────────────────────

def detect_task_type(y: pd.Series) -> str:
    """
    Heuristic: if target has ≤ 20 unique values and dtype is int/object → classification.
    Otherwise → regression.
    """
    unique_ratio = y.nunique() / len(y)
    if y.dtype == object or y.dtype.name == "category":
        return "classification"
    if y.nunique() <= 20 and unique_ratio < 0.05:
        return "classification"
    return "regression"


# ─────────────────────────────────────────────
# Data profiler
# ─────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """Compute a rich quality profile of the dataset."""

    X = df.drop(columns=[target_column])
    y = df[target_column]

    total_cells = df.shape[0] * df.shape[1]
    missing_total = int(df.isna().sum().sum())
    missing_pct   = round(missing_total / total_cells * 100, 2)

    # Duplicates
    duplicate_rows = int(df.duplicated().sum())

    # Outlier detection (IQR method on numeric columns)
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    outlier_counts: dict[str, int] = {}
    for col in numeric_cols:
        q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        iqr = q3 - q1
        mask = (X[col] < q1 - 1.5 * iqr) | (X[col] > q3 + 1.5 * iqr)
        n = int(mask.sum())
        if n > 0:
            outlier_counts[col] = n

    # Correlation with target (numeric features only)
    corr_with_target: dict[str, float] = {}
    if pd.api.types.is_numeric_dtype(y):
        for col in numeric_cols:
            c = df[col].corr(y)
            if not np.isnan(c):
                corr_with_target[col] = round(float(c), 4)
        corr_with_target = dict(
            sorted(corr_with_target.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        )

    # Class imbalance (classification only)
    class_dist: dict[str, int] | None = None
    imbalance_ratio: float | None = None
    task_type = detect_task_type(y)
    if task_type == "classification":
        vc = y.value_counts()
        class_dist = vc.to_dict()
        if len(vc) >= 2:
            imbalance_ratio = round(float(vc.iloc[0] / vc.iloc[-1]), 2)

    # Constant / near-constant features
    constant_features = [
        col for col in X.columns
        if X[col].nunique() <= 1
    ]

    # High-cardinality categoricals
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    high_card = [
        col for col in cat_cols
        if X[col].nunique() > 50
    ]

    return {
        "rows":                  int(df.shape[0]),
        "columns":               int(df.shape[1]),
        "numeric_features":      int(len(numeric_cols)),
        "categorical_features":  int(len(cat_cols)),
        "missing_total":         missing_total,
        "missing_pct":           missing_pct,
        "duplicate_rows":        duplicate_rows,
        "outlier_counts":        outlier_counts,
        "top_correlations":      corr_with_target,
        "class_distribution":    class_dist,
        "imbalance_ratio":       imbalance_ratio,
        "constant_features":     constant_features,
        "high_cardinality_cols": high_card,
        "task_type":             task_type,
    }


# ─────────────────────────────────────────────
# Preprocessing builder
# ─────────────────────────────────────────────

def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    transformers: list = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    return ColumnTransformer(transformers, remainder="drop")


# ─────────────────────────────────────────────
# Model trainer
# ─────────────────────────────────────────────

def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    """Train baseline + random forest and return rich evaluation metrics."""

    # Encode target for classification
    le = None
    if task_type == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None,
    )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    # Baseline (interpretable)
    if task_type == "regression":
        baseline_estimator = LinearRegression()
        rf_estimator       = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        baseline_estimator = LogisticRegression(max_iter=500, random_state=42)
        rf_estimator       = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    baseline_model = Pipeline([
        ("preprocessing", preprocessor),
        ("estimator",     baseline_estimator),
    ])
    rf_model = Pipeline([
        ("preprocessing", build_preprocessor(numeric_features, categorical_features)),
        ("estimator",     rf_estimator),
    ])

    baseline_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    y_pred_base = baseline_model.predict(X_test)
    y_pred_rf   = rf_model.predict(X_test)

    metrics: dict[str, Any] = {}

    if task_type == "regression":
        metrics["r2_baseline"]   = round(float(r2_score(y_test, y_pred_base)), 4)
        metrics["r2_rf"]         = round(float(r2_score(y_test, y_pred_rf)),   4)
        metrics["mae"]           = round(float(mean_absolute_error(y_test, y_pred_rf)), 4)
        metrics["rmse"]          = round(float(np.sqrt(mean_squared_error(y_test, y_pred_rf))), 4)

        # Cross-validation R²
        cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="r2")
        metrics["cv_r2_mean"]  = round(float(cv_scores.mean()), 4)
        metrics["cv_r2_std"]   = round(float(cv_scores.std()),  4)

    else:
        n_classes = y.nunique()
        avg = "binary" if n_classes == 2 else "weighted"

        metrics["accuracy_baseline"] = round(float(accuracy_score(y_test, y_pred_base)), 4)
        metrics["accuracy_rf"]       = round(float(accuracy_score(y_test, y_pred_rf)),   4)
        metrics["f1_score"]          = round(float(f1_score(y_test, y_pred_rf, average=avg, zero_division=0)), 4)

        # AUC-ROC (binary only)
        if n_classes == 2 and hasattr(rf_model, "predict_proba"):
            try:
                y_prob = rf_model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
            except Exception:
                pass

        # Cross-validation accuracy
        cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="accuracy")
        metrics["cv_accuracy_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_accuracy_std"]  = round(float(cv_scores.std()),  4)

    # Feature importance from RF
    feature_importance: dict[str, float] = {}
    try:
        rf_step = rf_model.named_steps["estimator"]
        prep    = rf_model.named_steps["preprocessing"]

        # Get feature names after preprocessing
        all_names: list[str] = []
        for name, _, cols in prep.transformers_:
            if name == "num":
                all_names.extend(cols)
            elif name == "cat":
                enc = prep.named_transformers_["cat"].named_steps["encoder"]
                all_names.extend(enc.get_feature_names_out(cols).tolist())

        importances = rf_step.feature_importances_
        fi_pairs = sorted(
            zip(all_names, importances),
            key=lambda x: x[1],
            reverse=True,
        )[:10]
        feature_importance = {k: round(float(v), 5) for k, v in fi_pairs}
    except Exception:
        pass

    return metrics, feature_importance


# ─────────────────────────────────────────────
# Dataset health score
# ─────────────────────────────────────────────

def compute_health_score(profile: dict, metrics: dict, task_type: str) -> int:
    score = 100

    # Missing data penalty
    score -= int(profile["missing_pct"] * 0.6)

    # Duplicate penalty
    dup_ratio = profile["duplicate_rows"] / max(profile["rows"], 1)
    score -= int(dup_ratio * 20)

    # Constant features penalty
    score -= len(profile["constant_features"]) * 5

    # High cardinality penalty
    score -= min(len(profile["high_cardinality_cols"]) * 3, 15)

    # Class imbalance penalty
    if profile["imbalance_ratio"] and profile["imbalance_ratio"] > 5:
        score -= 15
    elif profile["imbalance_ratio"] and profile["imbalance_ratio"] > 2:
        score -= 5

    # Model performance
    if task_type == "regression":
        r2 = metrics.get("r2_rf", metrics.get("r2_baseline", 0))
        if r2 < 0:      score -= 35
        elif r2 < 0.3:  score -= 20
        elif r2 < 0.6:  score -= 10
    else:
        acc = metrics.get("accuracy_rf", metrics.get("accuracy_baseline", 0))
        if acc < 0.5:   score -= 35
        elif acc < 0.65: score -= 20
        elif acc < 0.8:  score -= 10

    return max(0, min(100, score))


# ─────────────────────────────────────────────
# LLM analysis
# ─────────────────────────────────────────────

def _build_llm_prompt(profile: dict, metrics: dict, task_type: str, health_score: int) -> str:
    return f"""You are a senior ML engineer reviewing an automated dataset diagnostics report.
Analyse the results below and produce a concise expert assessment as a JSON array of bullet strings.

== Dataset Profile ==
{json.dumps(profile, indent=2)}

== Model Metrics ==
{json.dumps(metrics, indent=2)}

== Task Type ==
{task_type}

== Dataset Health Score ==
{health_score} / 100

Instructions:
- Return ONLY a valid JSON array of 6-8 short bullet strings (no Markdown, no preamble).
- Cover: data quality issues, model signal strength, key risks, actionable next steps.
- Be direct and specific — reference actual numbers from the report.
- Example format: ["The dataset has X rows ...", "R² of 0.72 suggests ...", ...]
"""


def generate_llm_analysis(
    profile: dict,
    metrics: dict,
    task_type: str,
    health_score: int,
    api_key: str | None = None,
) -> list[str]:
    """
    Try Anthropic Claude first, then fall back to rule-based analysis.
    Pass api_key=None to use os.environ['ANTHROPIC_API_KEY'] automatically.
    """
    key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")

    if key:
        try:
            import anthropic  # pip install anthropic
            client = anthropic.Anthropic(api_key=key)
            prompt = _build_llm_prompt(profile, metrics, task_type, health_score)

            message = client.messages.create(
                model="claude-sonnet-4-5",
                max_tokens=800,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = message.content[0].text.strip()

            # Strip Markdown fences if present
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])

            bullets = json.loads(raw)
            if isinstance(bullets, list):
                return [str(b) for b in bullets]
        except Exception:
            pass  # Fall through to rule-based

    return _rule_based_analysis(profile, metrics, task_type, health_score)


def _rule_based_analysis(
    profile: dict,
    metrics: dict,
    task_type: str,
    health_score: int,
) -> list[str]:
    insights: list[str] = []

    # Size
    insights.append(
        f"Dataset contains {profile['rows']:,} rows × {profile['columns']} columns "
        f"({profile['numeric_features']} numeric, {profile['categorical_features']} categorical features)."
    )

    # Missing values
    if profile["missing_pct"] > 10:
        insights.append(
            f"⚠️ High missing-value rate: {profile['missing_pct']}% of all cells — imputation or removal needed."
        )
    elif profile["missing_total"] > 0:
        insights.append(
            f"Minor missing data ({profile['missing_pct']}%) — median/mode imputation applied automatically."
        )
    else:
        insights.append("No missing values detected — clean feature matrix for modeling.")

    # Duplicates
    if profile["duplicate_rows"] > 0:
        insights.append(
            f"⚠️ {profile['duplicate_rows']} duplicate rows found — these should be removed before training."
        )

    # Outliers
    if profile["outlier_counts"]:
        top_col = max(profile["outlier_counts"], key=profile["outlier_counts"].get)
        insights.append(
            f"Outliers detected in {len(profile['outlier_counts'])} features "
            f"(worst: '{top_col}' with {profile['outlier_counts'][top_col]} outliers) — "
            f"consider robust scaling or capping."
        )

    # Class imbalance
    if profile["imbalance_ratio"] and profile["imbalance_ratio"] > 3:
        insights.append(
            f"⚠️ Class imbalance ratio {profile['imbalance_ratio']}:1 — "
            f"use SMOTE, class_weight='balanced', or collect more minority samples."
        )

    # Model signal
    if task_type == "regression":
        r2 = metrics.get("r2_rf", metrics.get("r2_baseline", 0))
        mae = metrics.get("mae", "N/A")
        cv  = metrics.get("cv_r2_mean", "N/A")
        if r2 >= 0.8:
            insights.append(
                f"✅ Strong predictive signal: RF R²={r2}, MAE={mae}, CV-R²={cv} — ready for advanced modeling."
            )
        elif r2 >= 0.5:
            insights.append(
                f"Moderate signal: RF R²={r2}, MAE={mae} — feature engineering or more data may boost performance."
            )
        else:
            insights.append(
                f"⚠️ Weak signal: RF R²={r2}, MAE={mae} — reconsider target definition or gather stronger features."
            )
    else:
        acc = metrics.get("accuracy_rf", metrics.get("accuracy_baseline", 0))
        f1  = metrics.get("f1_score", "N/A")
        auc = metrics.get("roc_auc", "N/A")
        cv  = metrics.get("cv_accuracy_mean", "N/A")
        if acc >= 0.85:
            insights.append(
                f"✅ Strong classification signal: Accuracy={acc}, F1={f1}, AUC={auc}, CV={cv}."
            )
        elif acc >= 0.65:
            insights.append(
                f"Moderate accuracy: {acc} (F1={f1}) — try ensemble methods or better feature selection."
            )
        else:
            insights.append(
                f"⚠️ Poor accuracy ({acc}) — check for label noise, class imbalance, or insufficient features."
            )

    # Constant features
    if profile["constant_features"]:
        insights.append(
            f"⚠️ {len(profile['constant_features'])} constant feature(s) found "
            f"({', '.join(profile['constant_features'][:3])}) — drop them before training."
        )

    # High cardinality
    if profile["high_cardinality_cols"]:
        insights.append(
            f"High-cardinality categorical columns detected: {profile['high_cardinality_cols']} — "
            f"consider target encoding or frequency encoding instead of one-hot."
        )

    # Health score verdict
    if health_score >= 80:
        insights.append(f"✅ Overall health score: {health_score}/100 — dataset is production-ready.")
    elif health_score >= 60:
        insights.append(f"Dataset health score: {health_score}/100 — usable with minor remediation.")
    else:
        insights.append(f"⚠️ Low health score: {health_score}/100 — significant data quality work required.")

    return insights


# ─────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────

def run_debugger_pipeline(
    df: pd.DataFrame,
    target_column: str | None = None,
    api_key: str | None = None,
) -> dict[str, Any]:
    """
    Full AutoML diagnostics pipeline.

    Parameters
    ----------
    df            : Input DataFrame
    target_column : Name of the label column (auto-detected if None)
    api_key       : Anthropic API key (uses env var if None)

    Returns
    -------
    dict with keys: metrics, diagnosis, llm_analysis, feature_importance,
                    profile, task_type, health_score
    """
    df = df.copy()

    # Minimum size check
    if df.shape[0] < 10 or df.shape[1] < 2:
        return {
            "metrics":           {},
            "diagnosis":         "Dataset too small.",
            "llm_analysis":      ["Dataset is too small for reliable ML diagnostics (need ≥ 10 rows, ≥ 2 columns)."],
            "feature_importance": {},
            "profile":           {},
            "task_type":         "unknown",
            "health_score":      0,
        }

    # Target column resolution
    if target_column is None or target_column not in df.columns:
        target_column = df.columns[-1]

    # Coerce target for regression, clean for classification
    task_hint = detect_task_type(df[target_column])
    if task_hint == "regression":
        df[target_column] = pd.to_numeric(df[target_column], errors="coerce")
    df = df.dropna(subset=[target_column])

    if df.shape[0] < 10:
        return {
            "metrics":           {},
            "diagnosis":         "Not enough valid target values after cleaning.",
            "llm_analysis":      ["Too many missing/invalid values in the target column — cannot train."],
            "feature_importance": {},
            "profile":           {},
            "task_type":         task_hint,
            "health_score":      0,
        }

    # Drop constant columns silently (they break pipelines)
    non_const = [c for c in df.columns if df[c].nunique() > 1 or c == target_column]
    df = df[non_const]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    task_type = detect_task_type(y)

    numeric_features     = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Profile
    try:
        profile = profile_dataset(df, target_column)
    except Exception:
        profile = {"rows": df.shape[0], "columns": df.shape[1]}

    # Train & evaluate
    try:
        metrics, feature_importance = train_and_evaluate(
            X, y, task_type, numeric_features, categorical_features
        )
    except Exception as e:
        return {
            "metrics":            {},
            "diagnosis":          f"Model training failed: {e}",
            "llm_analysis":       [f"Training error: {traceback.format_exc(limit=2)}"],
            "feature_importance": {},
            "profile":            profile,
            "task_type":          task_type,
            "health_score":       0,
        }

    # Add profile metadata into metrics for display
    metrics["rows"]                = profile["rows"]
    metrics["columns"]             = profile["columns"]
    metrics["numeric_features"]    = profile["numeric_features"]
    metrics["categorical_features"]= profile["categorical_features"]
    metrics["missing_values"]      = profile["missing_total"]

    # Health score
    health_score = compute_health_score(profile, metrics, task_type)
    metrics["dataset_health_score"] = health_score

    # Diagnosis string
    if task_type == "regression":
        r2 = metrics.get("r2_rf", 0)
        if r2 >= 0.8:    diagnosis = "Strong predictive signal detected — dataset is ML-ready."
        elif r2 >= 0.5:  diagnosis = "Moderate predictive signal — feature engineering recommended."
        elif r2 >= 0.0:  diagnosis = "Weak predictive signal — consider better features or more data."
        else:            diagnosis = "Very weak signal (negative R²) — major data quality issues."
    else:
        acc = metrics.get("accuracy_rf", 0)
        if acc >= 0.85:  diagnosis = "Strong classification performance — dataset is ML-ready."
        elif acc >= 0.65: diagnosis = "Moderate accuracy — further tuning needed."
        else:            diagnosis = "Low accuracy — review labels, features, and class balance."

    # LLM analysis
    llm_analysis = generate_llm_analysis(
        profile, metrics, task_type, health_score, api_key=api_key
    )

    return {
        "metrics":            metrics,
        "diagnosis":          diagnosis,
        "llm_analysis":       llm_analysis,
        "feature_importance": feature_importance,
        "profile":            profile,
        "task_type":          task_type,
        "health_score":       health_score,
    }