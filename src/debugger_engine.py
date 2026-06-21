"""
AutoML Debugger — Core Engine  (Stage 1 v3.0)
===============================================
Handles:
  - Task auto-detection with explanation
  - Preprocessing summary (what happens and WHY — shown to user before training)
  - Rich data profiling
  - Data leakage detection
  - Time-series detection with chronological split
  - Dataset health score with per-dimension breakdown
  - Dataset cleaning + export
  - Groq LLM analysis (rule-based fallback)
  - PDF report generation
"""

from __future__ import annotations

import io
import os
import json
import warnings
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# TASK AUTO-DETECTION  (with explanation text)
# ─────────────────────────────────────────────────────────────────

def detect_task_type(y: pd.Series) -> tuple[str, str]:
    """
    Returns (task_type, reason_string) so the UI can show the user
    exactly WHY a task type was chosen.
    """
    n_unique     = y.nunique()
    unique_ratio = n_unique / len(y)
    dtype_name   = str(y.dtype)

    if y.dtype == object or y.dtype.name == "category":
        reason = (
            f"Target column '{y.name}' has text/category values "
            f"({n_unique} unique classes) → Classification problem."
        )
        return "classification", reason

    if n_unique <= 2:
        reason = (
            f"Target column '{y.name}' has only {n_unique} unique numeric values "
            f"(binary) → Classification problem."
        )
        return "classification", reason

    if n_unique <= 20 and unique_ratio < 0.05:
        reason = (
            f"Target column '{y.name}' has {n_unique} unique values "
            f"({unique_ratio*100:.1f}% of rows) — low cardinality → Classification problem."
        )
        return "classification", reason

    reason = (
        f"Target column '{y.name}' is numeric with {n_unique} unique values "
        f"(dtype={dtype_name}, range={y.min():.2f}–{y.max():.2f}) → Regression problem."
    )
    return "regression", reason


# ─────────────────────────────────────────────────────────────────
# TIME-SERIES DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_timeseries(df: pd.DataFrame) -> dict[str, Any]:
    result = {
        "is_timeseries":   False,
        "datetime_column": None,
        "frequency_guess": None,
        "warnings":        [],
    }
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result["is_timeseries"]   = True
            result["datetime_column"] = col
            break
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], format="mixed", dayfirst=False)
                if parsed.notna().mean() > 0.9:
                    result["is_timeseries"]   = True
                    result["datetime_column"] = col
                    diffs  = parsed.dropna().sort_values().diff().dropna()
                    median = diffs.median()
                    days   = getattr(median, "days", 0)
                    result["frequency_guess"] = (
                        "Daily" if days == 1 else
                        "Weekly" if days == 7 else
                        "Monthly" if 28 <= days <= 31 else
                        "Sub-daily" if days == 0 else
                        f"~{days}-day intervals"
                    )
                    break
            except Exception:
                continue

    if result["is_timeseries"]:
        result["warnings"] = [
            f"⏱️ Time-series detected — column '{result['datetime_column']}' "
            f"({result['frequency_guess']}). "
            "Chronological train/test split will be used to prevent future-data leakage.",
            "⚠️ Standard random split is DISABLED — it would let future rows leak into training "
            "and artificially inflate all metrics.",
        ]
    return result


# ─────────────────────────────────────────────────────────────────
# PREPROCESSING SUMMARY  (shown to user BEFORE training)
# ─────────────────────────────────────────────────────────────────

def build_preprocessing_summary(
    df: pd.DataFrame,
    target_column: str,
    ts_datetime_col: str | None = None,
) -> list[dict[str, str]]:
    """
    Returns a list of dicts: [{action, column, reason, category}]
    Each entry explains one preprocessing decision in plain English.
    """
    X = df.drop(columns=[target_column])
    steps: list[dict[str, str]] = []

    # Datetime column dropped
    if ts_datetime_col and ts_datetime_col in X.columns:
        steps.append({
            "action":   "Drop from features",
            "column":   ts_datetime_col,
            "reason":   "Datetime column used only for chronological ordering — not a predictive feature.",
            "category": "time-series",
        })

    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # Missing values
    for col in numeric_cols:
        n = int(df[col].isna().sum())
        if n > 0:
            steps.append({
                "action":   f"Impute {n} missing values → median",
                "column":   col,
                "reason":   f"Numeric column has {n} nulls ({n/len(df)*100:.1f}%). Median is robust to outliers.",
                "category": "imputation",
            })
    for col in cat_cols:
        n = int(df[col].isna().sum())
        if n > 0:
            steps.append({
                "action":   f"Impute {n} missing values → most frequent",
                "column":   col,
                "reason":   f"Categorical column has {n} nulls ({n/len(df)*100:.1f}%). Mode fill preserves distribution.",
                "category": "imputation",
            })

    # Encoding
    for col in cat_cols:
        n_unique = df[col].nunique()
        # Detect date-like string columns and drop them (not useful as OHE features)
        is_date_like = False
        try:
            parsed = pd.to_datetime(df[col], format="mixed", dayfirst=False)
            if parsed.notna().mean() > 0.9:
                is_date_like = True
        except Exception:
            pass

        if is_date_like:
            steps.append({
                "action":   "Drop (date-like string column)",
                "column":   col,
                "reason":   "Detected as a date/time column. One-hot encoding dates creates meaningless sparse features. Use for time-series ordering only.",
                "category": "drop",
            })
        elif n_unique > 50:
            steps.append({
                "action":   f"Drop (high cardinality — {n_unique} unique values)",
                "column":   col,
                "reason":   f"One-hot encoding {n_unique} values would create too many sparse columns. Dropping.",
                "category": "encoding",
            })
        else:
            steps.append({
                "action":   f"One-Hot Encode → {n_unique} binary columns",
                "column":   col,
                "reason":   f"{n_unique} unique categories converted to binary indicators for ML compatibility.",
                "category": "encoding",
            })

    # Constant columns
    for col in X.columns:
        if X[col].nunique() <= 1:
            steps.append({
                "action":   "Drop (constant — zero variance)",
                "column":   col,
                "reason":   "This column has only one unique value. It carries no information for prediction.",
                "category": "drop",
            })

    # ID-like columns
    for col in X.columns:
        if (X[col].nunique() == len(X) and
                col.lower() in {"id", "index", "row_id", "uid", "uuid", "customer_id", "user_id"}):
            steps.append({
                "action":   "Drop (likely ID column)",
                "column":   col,
                "reason":   f"'{col}' has one unique value per row — it's an identifier, not a feature.",
                "category": "drop",
            })

    # Scaling
    if numeric_cols:
        steps.append({
            "action":   "StandardScaler (zero mean, unit variance)",
            "column":   f"{len(numeric_cols)} numeric column(s)",
            "reason":   "Scaling ensures Ridge/Logistic regression treat all features equally regardless of magnitude.",
            "category": "scaling",
        })

    return steps


# ─────────────────────────────────────────────────────────────────
# DATA LEAKAGE DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_leakage(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "leakage_candidates":        [],
        "high_correlation_features": {},
        "warnings":                  [],
    }
    y = df[target_column]
    if not pd.api.types.is_numeric_dtype(y):
        return result

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
                    f"🚨 '{col}' correlation={corr:.3f} with target — almost certainly derived from "
                    "or identical to the target. Verify before training."
                )
            elif corr > 0.85:
                result["warnings"].append(
                    f"⚠️ '{col}' correlation={corr:.3f} — high but may be legitimate. Verify."
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
    missing_by_col = {col: int(df[col].isna().sum()) for col in df.columns if df[col].isna().sum() > 0}

    duplicate_rows = int(df.duplicated().sum())

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols     = X.select_dtypes(exclude=[np.number]).columns

    outlier_counts: dict[str, int] = {}
    skewness:       dict[str, float] = {}
    for col in numeric_cols:
        q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        iqr    = q3 - q1
        mask   = (X[col] < q1 - 1.5 * iqr) | (X[col] > q3 + 1.5 * iqr)
        n      = int(mask.sum())
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

    task_type, _ = detect_task_type(y)
    class_dist:      dict | None  = None
    imbalance_ratio: float | None = None
    if task_type == "classification":
        vc            = y.value_counts()
        class_dist    = {str(k): int(v) for k, v in vc.items()}
        if len(vc) >= 2:
            imbalance_ratio = round(float(vc.iloc[0] / vc.iloc[-1]), 2)

    constant_features = [col for col in X.columns if X[col].nunique() <= 1]
    high_card         = [col for col in cat_cols if X[col].nunique() > 50]

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
        "task_type":              task_type,
    }


# ─────────────────────────────────────────────────────────────────
# HEALTH SCORE  (per-dimension breakdown)
# ─────────────────────────────────────────────────────────────────

def compute_health_score(
    profile: dict,
    best_metrics: dict,
    task_type: str,
    leakage: dict,
) -> dict[str, Any]:
    """
    Returns full breakdown dict, not just a number.
    Each dimension shows score, penalty, and reason.
    """
    dims: dict[str, dict] = {}

    # 1. Data Completeness (0-25)
    missing_pct = profile.get("missing_pct", 0)
    comp_score  = max(0, 25 - int(missing_pct * 0.5))
    dims["Data Completeness"] = {
        "score":   comp_score,
        "max":     25,
        "penalty": 25 - comp_score,
        "reason":  f"{missing_pct}% missing values" if missing_pct > 0 else "No missing values ✅",
    }

    # 2. Data Integrity (0-20)
    dup_ratio  = profile.get("duplicate_rows", 0) / max(profile.get("rows", 1), 1)
    const_pen  = min(len(profile.get("constant_features", [])) * 5, 10)
    integ_pen  = int(dup_ratio * 20) + const_pen
    integ_score = max(0, 20 - integ_pen)
    dims["Data Integrity"] = {
        "score":   integ_score,
        "max":     20,
        "penalty": 20 - integ_score,
        "reason":  (
            f"{profile.get('duplicate_rows', 0)} duplicate rows, "
            f"{len(profile.get('constant_features', []))} constant feature(s)"
            if integ_pen > 0 else "No duplicates or constant features ✅"
        ),
    }

    # 3. Signal Strength (0-30)
    if task_type == "regression":
        r2 = best_metrics.get("best_r2", best_metrics.get("r2", 0))
        sig_score = max(0, int(r2 * 30))
        sig_reason = f"Best model R²={r2:.3f}"
    else:
        acc = best_metrics.get("best_accuracy", best_metrics.get("accuracy", 0))
        sig_score  = max(0, int((acc - 0.5) / 0.5 * 30)) if acc > 0.5 else 0
        sig_reason = f"Best model accuracy={acc:.3f}"
    dims["Signal Strength"] = {
        "score":   sig_score,
        "max":     30,
        "penalty": 30 - sig_score,
        "reason":  sig_reason,
    }

    # 4. Leakage Risk (0-15)
    n_leaky = len(leakage.get("leakage_candidates", []))
    leak_score = max(0, 15 - n_leaky * 8)
    dims["Leakage Risk"] = {
        "score":   leak_score,
        "max":     15,
        "penalty": 15 - leak_score,
        "reason":  (
            f"{n_leaky} leakage candidate(s) detected 🚨"
            if n_leaky > 0 else "No leakage features detected ✅"
        ),
    }

    # 5. Class Balance (0-10, regression gets full marks)
    if task_type == "classification":
        ir = profile.get("imbalance_ratio") or 1
        if ir > 5:    bal_score, bal_reason = 0,  f"Severe imbalance {ir}:1 🚨"
        elif ir > 2:  bal_score, bal_reason = 5,  f"Moderate imbalance {ir}:1 ⚠️"
        else:         bal_score, bal_reason = 10, f"Balanced classes {ir}:1 ✅"
    else:
        bal_score  = 10
        bal_reason = "N/A for regression ✅"
    dims["Class Balance"] = {
        "score":   bal_score,
        "max":     10,
        "penalty": 10 - bal_score,
        "reason":  bal_reason,
    }

    total = sum(d["score"] for d in dims.values())

    if total >= 80:   verdict = "🟢 Production-Ready"
    elif total >= 60: verdict = "🟡 Needs Minor Work"
    else:             verdict = "🔴 Significant Issues"

    return {
        "total":      total,
        "verdict":    verdict,
        "dimensions": dims,
    }


# ─────────────────────────────────────────────────────────────────
# DATASET CLEANING
# ─────────────────────────────────────────────────────────────────

def clean_dataset(
    df: pd.DataFrame,
    target_column: str,
    remove_duplicates: bool = True,
    impute_missing:    bool = True,
    cap_outliers:      bool = True,
    drop_constant:     bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    report: dict[str, Any] = {"original_shape": df.shape, "actions": []}
    cleaned = df.copy()

    if drop_constant:
        cols = [c for c in cleaned.columns if c != target_column and cleaned[c].nunique() <= 1]
        if cols:
            cleaned.drop(columns=cols, inplace=True)
            report["actions"].append(f"Dropped {len(cols)} constant column(s): {cols}")

    if remove_duplicates:
        before = len(cleaned)
        cleaned.drop_duplicates(inplace=True)
        n = before - len(cleaned)
        if n:
            report["actions"].append(f"Removed {n} duplicate row(s).")

    if impute_missing:
        num_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = cleaned.select_dtypes(exclude=[np.number]).columns.tolist()
        fn = fc = 0
        for col in num_cols:
            n = cleaned[col].isna().sum()
            if n:
                cleaned[col].fillna(cleaned[col].median(), inplace=True)
                fn += n
        for col in cat_cols:
            n = cleaned[col].isna().sum()
            if n:
                cleaned[col].fillna(cleaned[col].mode()[0], inplace=True)
                fc += n
        if fn + fc > 0:
            report["actions"].append(
                f"Imputed {fn} numeric (median) + {fc} categorical (mode) missing values."
            )

    if cap_outliers:
        num_cols = cleaned.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in num_cols:
            num_cols.remove(target_column)
        capped = []
        for col in num_cols:
            q1, q3 = cleaned[col].quantile(0.25), cleaned[col].quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            if ((cleaned[col] < lo) | (cleaned[col] > hi)).sum() > 0:
                cleaned[col] = cleaned[col].clip(lo, hi)
                capped.append(col)
        if capped:
            report["actions"].append(
                f"Capped outliers (IQR) in {len(capped)} column(s): "
                f"{capped[:5]}{'...' if len(capped) > 5 else ''}"
            )

    report["final_shape"]   = cleaned.shape
    report["rows_removed"]  = report["original_shape"][0] - cleaned.shape[0]
    report["cols_removed"]  = report["original_shape"][1] - cleaned.shape[1]
    report["missing_after"] = int(cleaned.isna().sum().sum())
    return cleaned, report


# ─────────────────────────────────────────────────────────────────
# GROQ LLM ANALYSIS
# ─────────────────────────────────────────────────────────────────

def generate_llm_analysis(
    profile:      dict,
    model_results: dict,
    task_type:    str,
    health:       dict,
    leakage:      dict,
    ts_info:      dict,
    feature_importance: dict,
    preprocessing_steps: list,
    api_key:      str | None = None,
) -> dict[str, list[str]]:
    """
    Returns dict with keys:
      overall_analysis, feature_commentary, model_commentary, action_plan
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")
    if key:
        try:
            from groq import Groq
            client = Groq(api_key=key)

            prompt = f"""You are a senior ML engineer reviewing an AutoML pipeline report.
Return ONLY a valid JSON object with exactly these 4 keys (each is a list of strings):
{{
  "overall_analysis": [...],   // 4-5 bullets: dataset quality, key risks, overall verdict
  "feature_commentary": [...], // 3-4 bullets: what the top features mean, which are redundant, what to engineer
  "model_commentary": [...],   // 3-4 bullets: why winning model won, overfitting check, what to try next
  "action_plan": [...]         // 3-5 bullets: prioritized list of exact next steps before deployment
}}

DATA:
Task: {task_type}
Health Score: {health['total']}/100 ({health['verdict']})
Health Dimensions: {json.dumps({k: v['score'] for k,v in health['dimensions'].items()})}
Profile: rows={profile['rows']}, missing={profile['missing_pct']}%, duplicates={profile['duplicate_rows']}, outliers={len(profile['outlier_counts'])} cols
Leakage candidates: {leakage.get('leakage_candidates', [])}
Time-series: {ts_info.get('is_timeseries', False)} ({ts_info.get('frequency_guess', '')})
Model leaderboard: {json.dumps(model_results.get('leaderboard', [])[:3])}
Top features: {json.dumps(list(feature_importance.items())[:6])}
Imbalance ratio: {profile.get('imbalance_ratio', 'N/A')}

Rules: Be specific, reference actual numbers, no Markdown in strings, no preamble, pure JSON only."""

            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.3,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            parsed = json.loads(raw)
            if all(k in parsed for k in ["overall_analysis", "feature_commentary", "model_commentary", "action_plan"]):
                return parsed
        except Exception:
            pass

    return _rule_based_llm(profile, model_results, task_type, health, leakage, ts_info, feature_importance)


def _rule_based_llm(profile, model_results, task_type, health, leakage, ts_info, feature_importance):
    leaderboard = model_results.get("leaderboard", [])
    best = leaderboard[0] if leaderboard else {}

    # Overall
    overall = [
        f"Dataset: {profile['rows']:,} rows × {profile['columns']} columns "
        f"({profile['numeric_features']} numeric, {profile['categorical_features']} categorical). "
        f"Health score: {health['total']}/100 ({health['verdict']})."
    ]
    if profile["missing_pct"] > 10:
        overall.append(f"⚠️ {profile['missing_pct']}% missing values — imputation applied but review columns with >30% nulls for potential removal.")
    elif profile["missing_total"] > 0:
        overall.append(f"Minor missing data ({profile['missing_pct']}%) — median/mode imputation applied automatically.")
    else:
        overall.append("✅ Zero missing values — dataset is complete.")
    if leakage.get("leakage_candidates"):
        overall.append(f"🚨 {len(leakage['leakage_candidates'])} potential data leakage feature(s): {leakage['leakage_candidates']}. These must be verified before any deployment.")
    if ts_info.get("is_timeseries"):
        overall.append(f"⏱️ Time-series dataset ({ts_info['frequency_guess']}). Chronological split used. Standard random split would have leaked future data into training.")

    # Feature commentary
    feat = []
    if feature_importance:
        top3 = list(feature_importance.items())[:3]
        feat.append(f"Top 3 predictive features: {', '.join(f'{k} ({v:.3f})' for k,v in top3)}. These drive the majority of model decisions.")
        bottom = list(feature_importance.items())[-3:]
        feat.append(f"Lowest importance features: {', '.join(k for k,_ in bottom)}. Consider dropping them to reduce noise and overfitting risk.")
        feat.append("Features with importance < 0.01 are candidates for removal — they add complexity without predictive value.")
    else:
        feat.append("Feature importance not available for this model configuration.")
    corr = profile.get("top_correlations", {})
    if corr:
        top_corr = list(corr.items())[0]
        feat.append(f"'{top_corr[0]}' has the strongest linear correlation with target ({top_corr[1]:.3f}). This aligns with its high feature importance.")

    # Model commentary
    model_c = []
    if best:
        model_c.append(f"🏆 Best model: {best.get('model', '?')} with CV score={best.get('cv_score', '?'):.4f} — selected because it achieved highest cross-validated performance.")
        if len(leaderboard) > 1:
            second = leaderboard[1]
            model_c.append(f"Runner-up: {second.get('model', '?')} (CV={second.get('cv_score', '?'):.4f}) — gap of {abs(best.get('cv_score',0) - second.get('cv_score',0)):.4f} suggests {'clear winner' if abs(best.get('cv_score',0) - second.get('cv_score',0)) > 0.02 else 'similar performance — both worth considering'}.")
        train_cv_gap = abs(best.get("train_score", 0) - best.get("cv_score", 0))
        if train_cv_gap > 0.1:
            model_c.append(f"⚠️ Train/CV gap = {train_cv_gap:.3f} — possible overfitting. Consider reducing model complexity or adding regularization.")
        else:
            model_c.append(f"✅ Train/CV gap = {train_cv_gap:.3f} — model generalizes well, no significant overfitting detected.")
    else:
        model_c.append("No model results available.")

    # Action plan
    actions = []
    if leakage.get("leakage_candidates"):
        actions.append(f"1. CRITICAL: Investigate leakage features {leakage['leakage_candidates']} — drop or justify before training.")
    if profile["missing_pct"] > 20:
        actions.append("2. Review columns with >30% missing — imputation may introduce bias. Consider dropping them.")
    if profile.get("duplicate_rows", 0) > 0:
        actions.append(f"3. Remove {profile['duplicate_rows']} duplicate rows from the cleaned dataset before final training.")
    if profile.get("imbalance_ratio") and profile["imbalance_ratio"] > 3:
        actions.append(f"4. Address class imbalance ({profile['imbalance_ratio']}:1) — use SMOTE or class_weight='balanced'.")
    actions.append("5. Download the cleaned dataset and use it for your final model training.")
    actions.append("6. Retrain with XGBoost using the cleaned data and tune n_estimators + max_depth for best results.")

    return {
        "overall_analysis":   overall,
        "feature_commentary": feat,
        "model_commentary":   model_c,
        "action_plan":        actions,
    }


# ─────────────────────────────────────────────────────────────────
# PDF REPORT
# ─────────────────────────────────────────────────────────────────

def generate_pdf_report(
    profile:        dict,
    model_results:  dict,
    task_type:      str,
    health:         dict,
    llm_analysis:   dict,
    feature_importance: dict,
    leakage:        dict,
    ts_info:        dict,
    cleaning_report: dict | None = None,
) -> bytes:
    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import cm
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable

        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=2*cm, rightMargin=2*cm, topMargin=2*cm, bottomMargin=2*cm)

        styles   = getSampleStyleSheet()
        s_title  = ParagraphStyle("T",  fontSize=20, fontName="Helvetica-Bold",  textColor=colors.HexColor("#1a1f35"), spaceAfter=4)
        s_h2     = ParagraphStyle("H2", fontSize=13, fontName="Helvetica-Bold",  textColor=colors.HexColor("#2e3555"), spaceBefore=12, spaceAfter=4)
        s_body   = ParagraphStyle("B",  fontSize=9,  fontName="Helvetica",       leading=13, spaceAfter=3)
        s_bullet = ParagraphStyle("BU", fontSize=9,  fontName="Helvetica",       leading=13, leftIndent=10, spaceAfter=2)
        s_cap    = ParagraphStyle("C",  fontSize=7,  fontName="Helvetica-Oblique", textColor=colors.grey)

        DARK  = colors.HexColor("#1a1f35")
        BLUE  = colors.HexColor("#7eb6ff")
        STRIP = colors.HexColor("#f5f7ff")

        def table(data, col_w):
            t = Table(data, colWidths=col_w)
            t.setStyle(TableStyle([
                ("BACKGROUND",     (0,0), (-1,0), DARK),
                ("TEXTCOLOR",      (0,0), (-1,0), colors.white),
                ("FONTNAME",       (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",       (0,0), (-1,-1), 8),
                ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.white, STRIP]),
                ("GRID",           (0,0), (-1,-1), 0.4, colors.HexColor("#cccccc")),
                ("PADDING",        (0,0), (-1,-1), 5),
            ]))
            return t

        story = []
        story.append(Paragraph("🧠 AutoML Debugger — Diagnostic Report", s_title))
        story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')} | Task: {task_type.capitalize()} | Health: {health['total']}/100 {health['verdict']}", s_cap))
        story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceAfter=10))

        # Health breakdown
        story.append(Paragraph("Dataset Health Score", s_h2))
        hdata = [["Dimension", "Score", "Max", "Notes"]] + [
            [k, str(v["score"]), str(v["max"]), v["reason"]]
            for k, v in health["dimensions"].items()
        ] + [["TOTAL", str(health["total"]), "100", health["verdict"]]]
        story.append(table(hdata, [6*cm, 2*cm, 2*cm, 7*cm]))
        story.append(Spacer(1, 8))

        # Profile
        story.append(Paragraph("Dataset Profile", s_h2))
        pdata = [["Metric","Value"],["Rows", f"{profile['rows']:,}"],["Columns", str(profile['columns'])],
                 ["Missing", f"{profile['missing_pct']}%"],["Duplicates", str(profile['duplicate_rows'])],
                 ["Outlier cols", str(len(profile['outlier_counts']))],["Task", task_type]]
        story.append(table(pdata, [8*cm, 8*cm]))
        story.append(Spacer(1, 8))

        # Model leaderboard
        lb = model_results.get("leaderboard", [])
        if lb:
            story.append(Paragraph("Model Leaderboard", s_h2))
            metric_key = "cv_r2" if task_type == "regression" else "cv_accuracy"
            lbdata = [["Rank","Model","CV Score","Train Score","Status"]] + [
                [str(i+1), m.get("model","?"), f"{m.get('cv_score',0):.4f}",
                 f"{m.get('train_score',0):.4f}", "🏆 Best" if i==0 else ""]
                for i, m in enumerate(lb)
            ]
            story.append(table(lbdata, [1.5*cm, 5*cm, 3*cm, 3*cm, 3.5*cm]))
            story.append(Spacer(1, 8))

        # Expert analysis sections
        for section_key, section_title in [
            ("overall_analysis",   "Overall Analysis"),
            ("feature_commentary", "Feature Intelligence"),
            ("model_commentary",   "Model Commentary"),
            ("action_plan",        "Action Plan"),
        ]:
            bullets = llm_analysis.get(section_key, [])
            if bullets:
                story.append(Paragraph(section_title, s_h2))
                for b in bullets:
                    story.append(Paragraph(f"• {b}", s_bullet))
                story.append(Spacer(1, 6))

        # Leakage
        if leakage.get("leakage_candidates"):
            story.append(Paragraph("🚨 Data Leakage Risks", s_h2))
            for w in leakage.get("warnings", []):
                story.append(Paragraph(f"• {w}", s_bullet))

        # Cleaning summary
        if cleaning_report and cleaning_report.get("actions"):
            story.append(Paragraph("Dataset Cleaning Summary", s_h2))
            orig, final = cleaning_report["original_shape"], cleaning_report["final_shape"]
            story.append(Paragraph(f"Shape: {orig[0]}×{orig[1]} → {final[0]}×{final[1]}", s_body))
            for a in cleaning_report["actions"]:
                story.append(Paragraph(f"• {a}", s_bullet))

        story.append(HRFlowable(width="100%", thickness=1, color=BLUE, spaceBefore=10))
        story.append(Paragraph("AutoML Debugger v3.0 · Groq LLaMA 3.3 70B · scikit-learn · XGBoost · LightGBM", s_cap))

        doc.build(story)
        return buf.getvalue()

    except ImportError:
        lines = [f"AUTOML DEBUGGER REPORT | {datetime.now()}", f"Health: {health['total']}/100",
                 f"Task: {task_type}", "Action Plan:"] + llm_analysis.get("action_plan", [])
        return "\n".join(lines).encode("utf-8")
