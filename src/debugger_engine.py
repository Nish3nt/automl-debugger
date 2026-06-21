"""
AutoML Debugger Engine  (v2.0 — Industry-Ready)
================================================
Advancements over v1:
  1. Groq LLM API  (replaces Anthropic)
  2. Dataset cleaning + export
  3. Data leakage detection
  4. Time-series detection & safe train/test split
  5. PDF diagnostic report export (via reportlab)
"""

from __future__ import annotations

import io
import os
import json
import traceback
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score,
)

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# 1. TASK AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_task_type(y: pd.Series) -> str:
    unique_ratio = y.nunique() / len(y)
    if y.dtype == object or y.dtype.name == "category":
        return "classification"
    if y.nunique() <= 20 and unique_ratio < 0.05:
        return "classification"
    return "regression"


# ─────────────────────────────────────────────────────────────────
# ADVANCEMENT 4 — TIME-SERIES DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_timeseries(df: pd.DataFrame) -> dict[str, Any]:
    """
    Detect whether the dataset is a time-series.
    Returns metadata: is_timeseries, datetime_column, frequency_guess.
    """
    result = {
        "is_timeseries": False,
        "datetime_column": None,
        "frequency_guess": None,
        "warnings": [],
    }

    # Look for datetime-like columns
    for col in df.columns:
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], infer_datetime_format=True)
                # If most values parse successfully
                if parsed.notna().mean() > 0.9:
                    result["is_timeseries"] = True
                    result["datetime_column"] = col

                    # Guess frequency
                    diffs = parsed.dropna().sort_values().diff().dropna()
                    median_diff = diffs.median()
                    days = median_diff.days if hasattr(median_diff, "days") else 0
                    if days == 1:
                        result["frequency_guess"] = "Daily"
                    elif days == 7:
                        result["frequency_guess"] = "Weekly"
                    elif 28 <= days <= 31:
                        result["frequency_guess"] = "Monthly"
                    elif days == 0:
                        result["frequency_guess"] = "Sub-daily (intraday)"
                    else:
                        result["frequency_guess"] = f"~{days} day intervals"
                    break
            except Exception:
                continue
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            result["is_timeseries"] = True
            result["datetime_column"] = col
            break

    if result["is_timeseries"]:
        result["warnings"].append(
            f"⏱️ Time-series detected (column: '{result['datetime_column']}', "
            f"frequency: {result['frequency_guess']}). "
            "Using chronological train/test split to prevent data leakage from the future."
        )
        result["warnings"].append(
            "⚠️ Standard random train/test split is DISABLED for time-series data — "
            "it would leak future information into training and inflate all metrics."
        )

    return result


# ─────────────────────────────────────────────────────────────────
# ADVANCEMENT 3 — DATA LEAKAGE DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_leakage(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Flag features that are suspiciously correlated with the target.
    Correlation > 0.95 is almost always leakage in real datasets.
    """
    result = {
        "leakage_candidates": [],
        "high_correlation_features": {},
        "warnings": [],
    }

    y = df[target_column]
    if not pd.api.types.is_numeric_dtype(y):
        return result  # Can't compute correlation for non-numeric targets

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)

    for col in numeric_cols:
        try:
            corr = abs(df[col].corr(y))
            if np.isnan(corr):
                continue
            result["high_correlation_features"][col] = round(float(corr), 4)
            if corr > 0.95:
                result["leakage_candidates"].append(col)
                result["warnings"].append(
                    f"🚨 LEAKAGE RISK: '{col}' has correlation {corr:.3f} with target — "
                    "this feature likely contains future information or is derived from the target."
                )
            elif corr > 0.85:
                result["warnings"].append(
                    f"⚠️ High correlation: '{col}' ({corr:.3f}) — verify this isn't derived from the target."
                )
        except Exception:
            continue

    return result


# ─────────────────────────────────────────────────────────────────
# DATA PROFILER
# ─────────────────────────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    X = df.drop(columns=[target_column])
    y = df[target_column]

    total_cells   = df.shape[0] * df.shape[1]
    missing_total = int(df.isna().sum().sum())
    missing_pct   = round(missing_total / total_cells * 100, 2)
    missing_by_col = {
        col: int(df[col].isna().sum())
        for col in df.columns
        if df[col].isna().sum() > 0
    }

    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    outlier_counts: dict[str, int] = {}
    skewness: dict[str, float] = {}
    for col in numeric_cols:
        q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        iqr = q3 - q1
        mask = (X[col] < q1 - 1.5 * iqr) | (X[col] > q3 + 1.5 * iqr)
        n = int(mask.sum())
        if n > 0:
            outlier_counts[col] = n
        sk = float(X[col].skew())
        if not np.isnan(sk):
            skewness[col] = round(sk, 3)

    corr_with_target: dict[str, float] = {}
    if pd.api.types.is_numeric_dtype(y):
        for col in numeric_cols:
            c = df[col].corr(y)
            if not np.isnan(c):
                corr_with_target[col] = round(float(c), 4)
        corr_with_target = dict(
            sorted(corr_with_target.items(), key=lambda x: abs(x[1]), reverse=True)[:10]
        )

    class_dist: dict | None = None
    imbalance_ratio: float | None = None
    task_type = detect_task_type(y)
    if task_type == "classification":
        vc = y.value_counts()
        class_dist = {str(k): int(v) for k, v in vc.items()}
        if len(vc) >= 2:
            imbalance_ratio = round(float(vc.iloc[0] / vc.iloc[-1]), 2)

    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    cat_cols = X.select_dtypes(exclude=[np.number]).columns
    high_card = [col for col in cat_cols if X[col].nunique() > 50]

    # Column-level type warnings (NEW)
    type_warnings: list[str] = []
    for col in numeric_cols:
        uniq = X[col].dropna().unique()
        if set(uniq).issubset({0, 1, 0.0, 1.0}) and len(uniq) == 2:
            type_warnings.append(
                f"'{col}' has only values 0/1 but is numeric — may be a categorical flag."
            )
    for col in cat_cols:
        try:
            pd.to_datetime(X[col], infer_datetime_format=True)
            type_warnings.append(
                f"'{col}' looks like a date column — consider parsing it as datetime."
            )
        except Exception:
            pass
    # Detect potential ID columns
    for col in X.columns:
        if X[col].nunique() == len(X) and col.lower() in ["id", "index", "row_id", "uid", "uuid"]:
            type_warnings.append(
                f"'{col}' appears to be an ID column — should be excluded from features."
            )

    return {
        "rows":                   int(df.shape[0]),
        "columns":                int(df.shape[1]),
        "numeric_features":       int(len(numeric_cols)),
        "categorical_features":   int(len(cat_cols)),
        "missing_total":          missing_total,
        "missing_pct":            missing_pct,
        "missing_by_col":         missing_by_col,
        "duplicate_rows":         duplicate_rows,
        "outlier_counts":         outlier_counts,
        "skewness":               skewness,
        "top_correlations":       corr_with_target,
        "class_distribution":     class_dist,
        "imbalance_ratio":        imbalance_ratio,
        "constant_features":      constant_features,
        "high_cardinality_cols":  high_card,
        "type_warnings":          type_warnings,
        "task_type":              task_type,
    }


# ─────────────────────────────────────────────────────────────────
# ADVANCEMENT 2 — DATASET CLEANING
# ─────────────────────────────────────────────────────────────────

def clean_dataset(
    df: pd.DataFrame,
    target_column: str,
    remove_duplicates: bool = True,
    impute_missing: bool = True,
    cap_outliers: bool = True,
    drop_constant: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Clean the dataset and return (cleaned_df, cleaning_report).
    """
    report: dict[str, Any] = {
        "original_shape": df.shape,
        "actions": [],
    }

    cleaned = df.copy()

    # Step 1 — Drop constant features
    if drop_constant:
        const_cols = [
            col for col in cleaned.columns
            if col != target_column and cleaned[col].nunique() <= 1
        ]
        if const_cols:
            cleaned.drop(columns=const_cols, inplace=True)
            report["actions"].append(
                f"Dropped {len(const_cols)} constant column(s): {const_cols}"
            )

    # Step 2 — Remove duplicate rows
    if remove_duplicates:
        before = len(cleaned)
        cleaned.drop_duplicates(inplace=True)
        removed = before - len(cleaned)
        if removed:
            report["actions"].append(f"Removed {removed} duplicate row(s).")

    # Step 3 — Impute missing values
    if impute_missing:
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols     = cleaned.select_dtypes(exclude=[np.number]).columns.tolist()

        filled_num = 0
        for col in numeric_cols:
            n = cleaned[col].isna().sum()
            if n > 0:
                cleaned[col].fillna(cleaned[col].median(), inplace=True)
                filled_num += n

        filled_cat = 0
        for col in cat_cols:
            n = cleaned[col].isna().sum()
            if n > 0:
                cleaned[col].fillna(cleaned[col].mode()[0], inplace=True)
                filled_cat += n

        if filled_num + filled_cat > 0:
            report["actions"].append(
                f"Imputed {filled_num} numeric missing values (median) "
                f"and {filled_cat} categorical missing values (mode)."
            )

    # Step 4 — Cap outliers (IQR)
    if cap_outliers:
        numeric_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in numeric_cols:
            numeric_cols.remove(target_column)
        capped_cols = []
        for col in numeric_cols:
            q1 = cleaned[col].quantile(0.25)
            q3 = cleaned[col].quantile(0.75)
            iqr = q3 - q1
            lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_outliers = ((cleaned[col] < lower) | (cleaned[col] > upper)).sum()
            if n_outliers > 0:
                cleaned[col] = cleaned[col].clip(lower=lower, upper=upper)
                capped_cols.append(col)
        if capped_cols:
            report["actions"].append(
                f"Capped outliers (IQR method) in {len(capped_cols)} column(s): "
                f"{capped_cols[:5]}{'...' if len(capped_cols) > 5 else ''}"
            )

    report["final_shape"]     = cleaned.shape
    report["rows_removed"]    = report["original_shape"][0] - cleaned.shape[0]
    report["cols_removed"]    = report["original_shape"][1] - cleaned.shape[1]
    report["missing_after"]   = int(cleaned.isna().sum().sum())

    return cleaned, report


# ─────────────────────────────────────────────────────────────────
# PREPROCESSING + MODEL TRAINING
# ─────────────────────────────────────────────────────────────────

def build_preprocessor(numeric_features, categorical_features):
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))
    return ColumnTransformer(transformers, remainder="drop")


def train_and_evaluate(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    numeric_features: list[str],
    categorical_features: list[str],
    is_timeseries: bool = False,
) -> tuple[dict, dict]:
    le = None
    if task_type == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y), index=y.index, name=y.name)

    # ADVANCEMENT 4 — chronological split for time-series
    if is_timeseries:
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42,
            stratify=y if task_type == "classification" else None,
        )

    preprocessor = build_preprocessor(numeric_features, categorical_features)

    if task_type == "regression":
        baseline_estimator = LinearRegression()
        rf_estimator       = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    else:
        baseline_estimator = LogisticRegression(max_iter=500, random_state=42)
        rf_estimator       = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

    baseline_model = Pipeline([("preprocessing", preprocessor), ("estimator", baseline_estimator)])
    rf_model       = Pipeline([
        ("preprocessing", build_preprocessor(numeric_features, categorical_features)),
        ("estimator",     rf_estimator),
    ])

    baseline_model.fit(X_train, y_train)
    rf_model.fit(X_train, y_train)

    y_pred_base = baseline_model.predict(X_test)
    y_pred_rf   = rf_model.predict(X_test)

    metrics: dict[str, Any] = {}

    if task_type == "regression":
        metrics["r2_baseline"] = round(float(r2_score(y_test, y_pred_base)), 4)
        metrics["r2_rf"]       = round(float(r2_score(y_test, y_pred_rf)),   4)
        metrics["mae"]         = round(float(mean_absolute_error(y_test, y_pred_rf)), 4)
        metrics["rmse"]        = round(float(np.sqrt(mean_squared_error(y_test, y_pred_rf))), 4)

        if is_timeseries:
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring="r2")
        else:
            cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="r2")
        metrics["cv_r2_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_r2_std"]  = round(float(cv_scores.std()),  4)

    else:
        n_classes = y.nunique()
        avg = "binary" if n_classes == 2 else "weighted"

        metrics["accuracy_baseline"] = round(float(accuracy_score(y_test, y_pred_base)), 4)
        metrics["accuracy_rf"]       = round(float(accuracy_score(y_test, y_pred_rf)),   4)
        metrics["f1_score"]          = round(float(f1_score(y_test, y_pred_rf, average=avg, zero_division=0)), 4)

        if n_classes == 2 and hasattr(rf_model, "predict_proba"):
            try:
                y_prob = rf_model.predict_proba(X_test)[:, 1]
                metrics["roc_auc"] = round(float(roc_auc_score(y_test, y_prob)), 4)
            except Exception:
                pass

        if is_timeseries:
            tscv = TimeSeriesSplit(n_splits=5)
            cv_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring="accuracy")
        else:
            cv_scores = cross_val_score(rf_model, X, y, cv=5, scoring="accuracy")
        metrics["cv_accuracy_mean"] = round(float(cv_scores.mean()), 4)
        metrics["cv_accuracy_std"]  = round(float(cv_scores.std()),  4)

    # Feature importance
    feature_importance: dict[str, float] = {}
    try:
        rf_step  = rf_model.named_steps["estimator"]
        prep     = rf_model.named_steps["preprocessing"]
        all_names: list[str] = []
        for name, _, cols in prep.transformers_:
            if name == "num":
                all_names.extend(cols)
            elif name == "cat":
                enc = prep.named_transformers_["cat"].named_steps["encoder"]
                all_names.extend(enc.get_feature_names_out(cols).tolist())
        fi_pairs = sorted(
            zip(all_names, rf_step.feature_importances_),
            key=lambda x: x[1], reverse=True,
        )[:10]
        feature_importance = {k: round(float(v), 5) for k, v in fi_pairs}
    except Exception:
        pass

    return metrics, feature_importance


# ─────────────────────────────────────────────────────────────────
# HEALTH SCORE
# ─────────────────────────────────────────────────────────────────

def compute_health_score(profile: dict, metrics: dict, task_type: str) -> int:
    score = 100
    score -= int(profile.get("missing_pct", 0) * 0.6)
    dup_ratio = profile.get("duplicate_rows", 0) / max(profile.get("rows", 1), 1)
    score -= int(dup_ratio * 20)
    score -= len(profile.get("constant_features", [])) * 5
    score -= min(len(profile.get("high_cardinality_cols", [])) * 3, 15)
    ir = profile.get("imbalance_ratio")
    if ir and ir > 5:    score -= 15
    elif ir and ir > 2:  score -= 5

    if task_type == "regression":
        r2 = metrics.get("r2_rf", metrics.get("r2_baseline", 0))
        if r2 < 0:      score -= 35
        elif r2 < 0.3:  score -= 20
        elif r2 < 0.6:  score -= 10
    else:
        acc = metrics.get("accuracy_rf", metrics.get("accuracy_baseline", 0))
        if acc < 0.5:    score -= 35
        elif acc < 0.65: score -= 20
        elif acc < 0.8:  score -= 10

    return max(0, min(100, score))


# ─────────────────────────────────────────────────────────────────
# ADVANCEMENT 1 — GROQ LLM ANALYSIS
# ─────────────────────────────────────────────────────────────────

def _build_llm_prompt(
    profile: dict,
    metrics: dict,
    task_type: str,
    health_score: int,
    leakage: dict,
    ts_info: dict,
) -> str:
    leakage_note = ""
    if leakage.get("leakage_candidates"):
        leakage_note = f"\n== DATA LEAKAGE CANDIDATES ==\n{leakage['leakage_candidates']}\n"
    ts_note = ""
    if ts_info.get("is_timeseries"):
        ts_note = f"\n== TIME-SERIES DETECTED ==\nDatetime column: {ts_info['datetime_column']}, Frequency: {ts_info['frequency_guess']}\n"

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
{leakage_note}{ts_note}
Instructions:
- Return ONLY a valid JSON array of 6-8 short bullet strings (no Markdown, no preamble).
- Cover: data quality issues, model signal strength, leakage risks, time-series concerns (if any), actionable next steps.
- Be direct and specific — reference actual numbers from the report.
- Example format: ["The dataset has X rows ...", "R² of 0.72 suggests ...", ...]
"""


def generate_llm_analysis(
    profile: dict,
    metrics: dict,
    task_type: str,
    health_score: int,
    leakage: dict,
    ts_info: dict,
    api_key: str | None = None,
) -> list[str]:
    """Try Groq first, then fall back to rule-based analysis."""
    key = api_key or os.environ.get("GROQ_API_KEY", "")

    if key:
        try:
            from groq import Groq
            client = Groq(api_key=key)
            prompt = _build_llm_prompt(profile, metrics, task_type, health_score, leakage, ts_info)

            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.3,
            )
            raw = response.choices[0].message.content.strip()

            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])

            bullets = json.loads(raw)
            if isinstance(bullets, list):
                return [str(b) for b in bullets]
        except Exception:
            pass  # Fall through to rule-based

    return _rule_based_analysis(profile, metrics, task_type, health_score, leakage, ts_info)


def _rule_based_analysis(
    profile: dict,
    metrics: dict,
    task_type: str,
    health_score: int,
    leakage: dict,
    ts_info: dict,
) -> list[str]:
    insights: list[str] = []

    insights.append(
        f"Dataset contains {profile.get('rows', 0):,} rows × {profile.get('columns', 0)} columns "
        f"({profile.get('numeric_features', 0)} numeric, {profile.get('categorical_features', 0)} categorical)."
    )

    # Time-series
    if ts_info.get("is_timeseries"):
        insights.append(
            f"⏱️ Time-series dataset detected ('{ts_info['datetime_column']}', {ts_info['frequency_guess']}). "
            "Chronological split used — future leakage prevented."
        )

    # Leakage
    if leakage.get("leakage_candidates"):
        insights.append(
            f"🚨 {len(leakage['leakage_candidates'])} potential data leakage feature(s) found: "
            f"{leakage['leakage_candidates']} — correlation > 0.95 with target. Verify before deploying."
        )

    # Missing values
    if profile.get("missing_pct", 0) > 10:
        insights.append(f"⚠️ High missing-value rate: {profile['missing_pct']}% — imputation or removal needed.")
    elif profile.get("missing_total", 0) > 0:
        insights.append(f"Minor missing data ({profile['missing_pct']}%) — median/mode imputation applied.")
    else:
        insights.append("✅ No missing values — clean feature matrix.")

    # Duplicates
    if profile.get("duplicate_rows", 0) > 0:
        insights.append(f"⚠️ {profile['duplicate_rows']} duplicate rows — remove before training.")

    # Outliers
    if profile.get("outlier_counts"):
        top_col = max(profile["outlier_counts"], key=profile["outlier_counts"].get)
        insights.append(
            f"Outliers in {len(profile['outlier_counts'])} features "
            f"(worst: '{top_col}' with {profile['outlier_counts'][top_col]}) — consider IQR capping."
        )

    # Class imbalance
    if profile.get("imbalance_ratio") and profile["imbalance_ratio"] > 3:
        insights.append(
            f"⚠️ Class imbalance {profile['imbalance_ratio']}:1 — use SMOTE or class_weight='balanced'."
        )

    # Model signal
    if task_type == "regression":
        r2  = metrics.get("r2_rf", metrics.get("r2_baseline", 0))
        mae = metrics.get("mae", "N/A")
        cv  = metrics.get("cv_r2_mean", "N/A")
        if r2 >= 0.8:
            insights.append(f"✅ Strong signal: RF R²={r2}, MAE={mae}, CV-R²={cv} — ready for advanced modeling.")
        elif r2 >= 0.5:
            insights.append(f"Moderate signal: RF R²={r2}, MAE={mae} — feature engineering may help.")
        else:
            insights.append(f"⚠️ Weak signal: RF R²={r2} — reconsider target or add stronger features.")
    else:
        acc = metrics.get("accuracy_rf", metrics.get("accuracy_baseline", 0))
        f1  = metrics.get("f1_score", "N/A")
        auc = metrics.get("roc_auc", "N/A")
        if acc >= 0.85:
            insights.append(f"✅ Strong classification: Accuracy={acc}, F1={f1}, AUC={auc}.")
        elif acc >= 0.65:
            insights.append(f"Moderate accuracy: {acc} (F1={f1}) — try ensemble methods.")
        else:
            insights.append(f"⚠️ Poor accuracy ({acc}) — check label noise or class imbalance.")

    # Health verdict
    if health_score >= 80:
        insights.append(f"✅ Health score: {health_score}/100 — dataset is production-ready.")
    elif health_score >= 60:
        insights.append(f"Health score: {health_score}/100 — usable with minor remediation.")
    else:
        insights.append(f"⚠️ Health score: {health_score}/100 — significant data quality work required.")

    return insights


# ─────────────────────────────────────────────────────────────────
# ADVANCEMENT 5 — PDF REPORT GENERATION
# ─────────────────────────────────────────────────────────────────

def generate_pdf_report(
    profile: dict,
    metrics: dict,
    task_type: str,
    health_score: int,
    diagnosis: str,
    llm_analysis: list[str],
    feature_importance: dict,
    leakage: dict,
    ts_info: dict,
    cleaning_report: dict | None = None,
) -> bytes:
    """Generate a PDF diagnostic report and return as bytes."""
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
        )

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer, pagesize=A4,
            leftMargin=2*cm, rightMargin=2*cm,
            topMargin=2*cm, bottomMargin=2*cm,
        )

        styles = getSampleStyleSheet()
        style_title   = ParagraphStyle("Title",   fontSize=22, fontName="Helvetica-Bold",  textColor=colors.HexColor("#1a1f35"), spaceAfter=6)
        style_h2      = ParagraphStyle("H2",      fontSize=14, fontName="Helvetica-Bold",  textColor=colors.HexColor("#2e3555"), spaceBefore=14, spaceAfter=4)
        style_body    = ParagraphStyle("Body",    fontSize=10, fontName="Helvetica",       leading=14, spaceAfter=4)
        style_bullet  = ParagraphStyle("Bullet",  fontSize=10, fontName="Helvetica",       leading=14, leftIndent=12, spaceAfter=3)
        style_caption = ParagraphStyle("Caption", fontSize=8,  fontName="Helvetica-Oblique", textColor=colors.grey)

        BLUE   = colors.HexColor("#7eb6ff")
        DARK   = colors.HexColor("#1a1f35")
        GREEN  = colors.HexColor("#5dbc8a")
        RED    = colors.HexColor("#e07070")
        AMBER  = colors.HexColor("#e0b070")

        story = []

        # Header
        story.append(Paragraph("🧠 AutoML Debugger — Diagnostic Report", style_title))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style_caption))
        story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=12))

        # Health Score
        story.append(Paragraph("Dataset Health Score", style_h2))
        hs_color = GREEN if health_score >= 80 else (AMBER if health_score >= 60 else RED)
        verdict  = "Production-Ready" if health_score >= 80 else ("Needs Minor Work" if health_score >= 60 else "Significant Issues")
        story.append(Paragraph(f"<font color='#{hs_color.hexval()[2:]}' size='18'><b>{health_score} / 100</b></font> — {verdict}", style_body))
        story.append(Paragraph(f"<b>Diagnosis:</b> {diagnosis}", style_body))
        story.append(Spacer(1, 8))

        # Dataset Profile
        story.append(Paragraph("Dataset Profile", style_h2))
        profile_data = [
            ["Metric", "Value"],
            ["Rows",              f"{profile.get('rows', 0):,}"],
            ["Columns",           str(profile.get('columns', 0))],
            ["Numeric Features",  str(profile.get('numeric_features', 0))],
            ["Categorical Features", str(profile.get('categorical_features', 0))],
            ["Missing Values",    f"{profile.get('missing_total', 0)} ({profile.get('missing_pct', 0)}%)"],
            ["Duplicate Rows",    str(profile.get('duplicate_rows', 0))],
            ["Task Type",         task_type.capitalize()],
            ["Constant Features", str(len(profile.get('constant_features', [])))],
            ["High-Cardinality Cols", str(len(profile.get('high_cardinality_cols', [])))],
        ]
        t = Table(profile_data, colWidths=[8*cm, 8*cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), DARK),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7ff")]),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("PADDING",     (0,0), (-1,-1), 6),
        ]))
        story.append(t)
        story.append(Spacer(1, 10))

        # Time-series info
        if ts_info.get("is_timeseries"):
            story.append(Paragraph("⏱️ Time-Series Detected", style_h2))
            story.append(Paragraph(
                f"DateTime column: <b>{ts_info['datetime_column']}</b> | "
                f"Frequency: <b>{ts_info['frequency_guess']}</b>. "
                "Chronological train/test split was applied to prevent future data leakage.",
                style_body
            ))
            story.append(Spacer(1, 6))

        # Data leakage
        if leakage.get("leakage_candidates"):
            story.append(Paragraph("🚨 Data Leakage Risks", style_h2))
            for w in leakage.get("warnings", []):
                story.append(Paragraph(f"• {w}", style_bullet))
            story.append(Spacer(1, 6))

        # Model Metrics
        story.append(Paragraph("Model Performance Metrics", style_h2))
        if task_type == "regression":
            metric_data = [
                ["Metric", "Value"],
                ["R² (Baseline)",  str(metrics.get("r2_baseline", "—"))],
                ["R² (Random Forest)", str(metrics.get("r2_rf", "—"))],
                ["MAE",            str(metrics.get("mae", "—"))],
                ["RMSE",           str(metrics.get("rmse", "—"))],
                ["CV R² Mean",     str(metrics.get("cv_r2_mean", "—"))],
                ["CV R² Std",      str(metrics.get("cv_r2_std", "—"))],
            ]
        else:
            metric_data = [
                ["Metric", "Value"],
                ["Accuracy (Baseline)", str(metrics.get("accuracy_baseline", "—"))],
                ["Accuracy (RF)",       str(metrics.get("accuracy_rf", "—"))],
                ["F1 Score",            str(metrics.get("f1_score", "—"))],
                ["ROC-AUC",             str(metrics.get("roc_auc", "—"))],
                ["CV Accuracy Mean",    str(metrics.get("cv_accuracy_mean", "—"))],
                ["CV Accuracy Std",     str(metrics.get("cv_accuracy_std", "—"))],
            ]
        mt = Table(metric_data, colWidths=[8*cm, 8*cm])
        mt.setStyle(TableStyle([
            ("BACKGROUND",  (0,0), (-1,0), DARK),
            ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
            ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
            ("FONTSIZE",    (0,0), (-1,-1), 9),
            ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7ff")]),
            ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
            ("PADDING",     (0,0), (-1,-1), 6),
        ]))
        story.append(mt)
        story.append(Spacer(1, 10))

        # Expert Analysis
        story.append(Paragraph("Expert Analysis (AI-Generated)", style_h2))
        for bullet in llm_analysis:
            story.append(Paragraph(f"• {bullet}", style_bullet))
        story.append(Spacer(1, 10))

        # Feature Importance
        if feature_importance:
            story.append(Paragraph("Top Feature Importance (Random Forest)", style_h2))
            fi_data = [["Feature", "Importance"]] + [
                [k, f"{v:.5f}"] for k, v in list(feature_importance.items())[:8]
            ]
            fi_table = Table(fi_data, colWidths=[10*cm, 6*cm])
            fi_table.setStyle(TableStyle([
                ("BACKGROUND",  (0,0), (-1,0), DARK),
                ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
                ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",    (0,0), (-1,-1), 9),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, colors.HexColor("#f5f7ff")]),
                ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#cccccc")),
                ("PADDING",     (0,0), (-1,-1), 6),
            ]))
            story.append(fi_table)
            story.append(Spacer(1, 10))

        # Cleaning report
        if cleaning_report:
            story.append(Paragraph("Dataset Cleaning Summary", style_h2))
            orig = cleaning_report.get("original_shape", ("?", "?"))
            final = cleaning_report.get("final_shape", ("?", "?"))
            story.append(Paragraph(
                f"Original shape: <b>{orig[0]} × {orig[1]}</b> → "
                f"Cleaned shape: <b>{final[0]} × {final[1]}</b>",
                style_body
            ))
            for action in cleaning_report.get("actions", []):
                story.append(Paragraph(f"• {action}", style_bullet))
            story.append(Spacer(1, 6))

        # Footer
        story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceBefore=12))
        story.append(Paragraph(
            "AutoML Debugger v2.0 · Powered by Groq LLaMA 3.3 70B · Built with scikit-learn & Streamlit",
            style_caption
        ))

        doc.build(story)
        return buffer.getvalue()

    except ImportError:
        # If reportlab not installed, return a plain-text fallback as bytes
        lines = [
            "AUTOML DEBUGGER — DIAGNOSTIC REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Health Score: {health_score}/100",
            f"Task Type: {task_type}",
            f"Diagnosis: {diagnosis}",
            "",
            "METRICS:",
            json.dumps(metrics, indent=2),
            "",
            "EXPERT ANALYSIS:",
        ] + [f"• {b}" for b in llm_analysis]
        return "\n".join(lines).encode("utf-8")


# ─────────────────────────────────────────────────────────────────
# MAIN PIPELINE ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def run_debugger_pipeline(
    df: pd.DataFrame,
    target_column: str | None = None,
    api_key: str | None = None,
    run_cleaning: bool = False,
    cleaning_options: dict | None = None,
) -> dict[str, Any]:
    """
    Full AutoML diagnostics pipeline (v2.0).

    Parameters
    ----------
    df               : Input DataFrame
    target_column    : Label column (auto-detected if None)
    api_key          : Groq API key (uses GROQ_API_KEY env var if None)
    run_cleaning     : Whether to run and return a cleaned dataset
    cleaning_options : Dict of cleaning flags (see clean_dataset)

    Returns
    -------
    dict with keys:
        metrics, diagnosis, llm_analysis, feature_importance,
        profile, task_type, health_score,
        leakage, ts_info,
        cleaned_df (if run_cleaning=True), cleaning_report,
        pdf_bytes
    """
    df = df.copy()

    if df.shape[0] < 10 or df.shape[1] < 2:
        return {
            "metrics": {}, "diagnosis": "Dataset too small.",
            "llm_analysis": ["Dataset is too small for reliable ML diagnostics (need ≥ 10 rows, ≥ 2 columns)."],
            "feature_importance": {}, "profile": {}, "task_type": "unknown",
            "health_score": 0, "leakage": {}, "ts_info": {},
            "cleaned_df": None, "cleaning_report": {}, "pdf_bytes": None,
        }

    # Target resolution
    if target_column is None or target_column not in df.columns:
        target_column = df.columns[-1]

    # ADVANCEMENT 4 — Time-series detection (before any splitting)
    ts_info = detect_timeseries(df)

    # Drop the datetime column from features if detected
    df_model = df.copy()
    if ts_info["is_timeseries"] and ts_info["datetime_column"]:
        dt_col = ts_info["datetime_column"]
        if dt_col != target_column and dt_col in df_model.columns:
            df_model = df_model.drop(columns=[dt_col])

    # Coerce target
    task_hint = detect_task_type(df_model[target_column])
    if task_hint == "regression":
        df_model[target_column] = pd.to_numeric(df_model[target_column], errors="coerce")
    df_model = df_model.dropna(subset=[target_column])

    if df_model.shape[0] < 10:
        return {
            "metrics": {}, "diagnosis": "Not enough valid target values after cleaning.",
            "llm_analysis": ["Too many missing/invalid values in the target column."],
            "feature_importance": {}, "profile": {}, "task_type": task_hint,
            "health_score": 0, "leakage": {}, "ts_info": ts_info,
            "cleaned_df": None, "cleaning_report": {}, "pdf_bytes": None,
        }

    # Drop constant columns
    non_const = [c for c in df_model.columns if df_model[c].nunique() > 1 or c == target_column]
    df_model = df_model[non_const]

    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]
    task_type = detect_task_type(y)

    numeric_features     = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Profile
    try:
        profile = profile_dataset(df_model, target_column)
    except Exception:
        profile = {"rows": df_model.shape[0], "columns": df_model.shape[1]}

    # ADVANCEMENT 3 — Leakage detection
    leakage = detect_leakage(df_model, target_column)

    # Train & evaluate
    try:
        metrics, feature_importance = train_and_evaluate(
            X, y, task_type, numeric_features, categorical_features,
            is_timeseries=ts_info["is_timeseries"],
        )
    except Exception as e:
        return {
            "metrics": {}, "diagnosis": f"Model training failed: {e}",
            "llm_analysis": [f"Training error: {traceback.format_exc(limit=2)}"],
            "feature_importance": {}, "profile": profile, "task_type": task_type,
            "health_score": 0, "leakage": leakage, "ts_info": ts_info,
            "cleaned_df": None, "cleaning_report": {}, "pdf_bytes": None,
        }

    # Merge profile metadata into metrics
    metrics.update({
        "rows": profile["rows"], "columns": profile["columns"],
        "numeric_features": profile["numeric_features"],
        "categorical_features": profile["categorical_features"],
        "missing_values": profile["missing_total"],
    })

    health_score = compute_health_score(profile, metrics, task_type)
    metrics["dataset_health_score"] = health_score

    # Diagnosis string
    if task_type == "regression":
        r2 = metrics.get("r2_rf", 0)
        if r2 >= 0.8:    diagnosis = "Strong predictive signal — dataset is ML-ready."
        elif r2 >= 0.5:  diagnosis = "Moderate signal — feature engineering recommended."
        elif r2 >= 0.0:  diagnosis = "Weak signal — consider better features or more data."
        else:            diagnosis = "Very weak signal (negative R²) — major data quality issues."
    else:
        acc = metrics.get("accuracy_rf", 0)
        if acc >= 0.85:   diagnosis = "Strong classification performance — dataset is ML-ready."
        elif acc >= 0.65: diagnosis = "Moderate accuracy — further tuning needed."
        else:             diagnosis = "Low accuracy — review labels, features, and class balance."

    # LLM analysis
    llm_analysis = generate_llm_analysis(
        profile, metrics, task_type, health_score,
        leakage, ts_info, api_key=api_key,
    )

    # ADVANCEMENT 2 — Dataset cleaning
    cleaned_df     = None
    cleaning_report = {}
    if run_cleaning:
        opts = cleaning_options or {}
        cleaned_df, cleaning_report = clean_dataset(
            df,  # clean the original df (with datetime col)
            target_column,
            remove_duplicates=opts.get("remove_duplicates", True),
            impute_missing=opts.get("impute_missing", True),
            cap_outliers=opts.get("cap_outliers", True),
            drop_constant=opts.get("drop_constant", True),
        )

    # ADVANCEMENT 5 — PDF report
    pdf_bytes = generate_pdf_report(
        profile, metrics, task_type, health_score, diagnosis,
        llm_analysis, feature_importance, leakage, ts_info, cleaning_report or None,
    )

    return {
        "metrics":            metrics,
        "diagnosis":          diagnosis,
        "llm_analysis":       llm_analysis,
        "feature_importance": feature_importance,
        "profile":            profile,
        "task_type":          task_type,
        "health_score":       health_score,
        "leakage":            leakage,
        "ts_info":            ts_info,
        "cleaned_df":         cleaned_df,
        "cleaning_report":    cleaning_report,
        "pdf_bytes":          pdf_bytes,
    }
