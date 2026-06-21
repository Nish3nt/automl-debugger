"""
AutoML Debugger — Explainability & Analysis  (Stage 2 v3.0)
=============================================================
  1. SHAP explainability (global + local per-prediction)
  2. Residual analysis for regression
  3. Confusion matrix + plain English interpretation for classification
  4. Train/test distribution drift detection
  5. Model card generation
  6. Model save/load via joblib
"""

from __future__ import annotations

import io
import json
import warnings
import numpy as np
import pandas as pd
from typing import Any

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# 1. SHAP EXPLAINABILITY
# ─────────────────────────────────────────────────────────────────

def compute_shap_values(
    best_pipeline,
    X_sample: pd.DataFrame,
    numeric_features: list[str],
    categorical_features: list[str],
    max_samples: int = 200,
) -> dict[str, Any]:
    """
    Computes SHAP values using TreeExplainer (XGBoost/LightGBM)
    or KernelExplainer fallback for linear models.
    Returns global mean |SHAP| per feature + sample-level values.
    """
    try:
        import shap

        estimator = best_pipeline.named_steps["estimator"]
        prep      = best_pipeline.named_steps["preprocessing"]

        # Transform features
        X_t = prep.transform(X_sample.head(max_samples))

        # Get feature names after preprocessing
        feature_names: list[str] = []
        for name, _, cols in prep.transformers_:
            if name == "num":
                feature_names.extend(cols)
            elif name == "cat":
                enc = prep.named_transformers_["cat"].named_steps["encoder"]
                feature_names.extend(enc.get_feature_names_out(cols).tolist())

        X_transformed = pd.DataFrame(X_t, columns=feature_names)

        # Pick explainer
        model_name = type(estimator).__name__
        if "XGB" in model_name or "LGBM" in model_name or "LightGBM" in model_name:
            explainer  = shap.TreeExplainer(estimator)
            shap_vals  = explainer.shap_values(X_t)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[1]  # binary classification: class 1
        else:
            # Linear model — use LinearExplainer
            explainer = shap.LinearExplainer(estimator, X_t, feature_perturbation="interventional")
            shap_vals = explainer.shap_values(X_t)
            if isinstance(shap_vals, list):
                shap_vals = shap_vals[0]

        # Global mean |SHAP| per feature
        mean_abs = np.abs(shap_vals).mean(axis=0)
        global_importance = dict(
            sorted(zip(feature_names, mean_abs.tolist()), key=lambda x: x[1], reverse=True)[:15]
        )

        # Per-sample SHAP (first 5 rows for local explanations)
        sample_shap: list[dict] = []
        for i in range(min(5, len(shap_vals))):
            row_shap = dict(zip(feature_names, shap_vals[i].tolist()))
            # Top 5 contributors for this row
            top5 = sorted(row_shap.items(), key=lambda x: abs(x[1]), reverse=True)[:5]
            sample_shap.append({
                "row":          i,
                "top_features": [{"feature": k, "shap": round(v, 4)} for k, v in top5],
            })

        return {
            "available":          True,
            "global_importance":  global_importance,
            "sample_explanations": sample_shap,
            "feature_names":      feature_names,
            "shap_matrix":        shap_vals[:50].tolist(),  # first 50 rows for plotting
        }

    except Exception as e:
        return {"available": False, "error": str(e)}


# ─────────────────────────────────────────────────────────────────
# 2. RESIDUAL ANALYSIS  (regression only)
# ─────────────────────────────────────────────────────────────────

def compute_residual_analysis(
    y_true: pd.Series,
    y_pred: np.ndarray,
    model_name: str,
) -> dict[str, Any]:
    """
    Analyses residuals for systematic errors.
    Returns residual stats, pattern flags, and plain-English interpretation.
    """
    residuals  = np.array(y_true) - np.array(y_pred)
    abs_resid  = np.abs(residuals)
    pct_resid  = abs_resid / (np.abs(np.array(y_true)) + 1e-8) * 100

    # Basic stats
    stats = {
        "mean_residual":   round(float(residuals.mean()), 4),
        "std_residual":    round(float(residuals.std()),  4),
        "mae":             round(float(abs_resid.mean()), 4),
        "max_error":       round(float(abs_resid.max()),  4),
        "median_abs_pct":  round(float(np.median(pct_resid)), 2),
        "within_10pct":    round(float((pct_resid < 10).mean() * 100), 1),
        "within_20pct":    round(float((pct_resid < 20).mean() * 100), 1),
    }

    # Detect systematic bias
    patterns: list[str] = []
    interpretations: list[str] = []

    # Bias check
    if abs(stats["mean_residual"]) > stats["std_residual"] * 0.3:
        direction = "overestimates" if stats["mean_residual"] < 0 else "underestimates"
        patterns.append("systematic_bias")
        interpretations.append(
            f"⚠️ Systematic bias detected: model consistently {direction} "
            f"(mean residual = {stats['mean_residual']:.4f}). "
            "Consider adding bias correction or checking for missing features."
        )
    else:
        interpretations.append(
            f"✅ No systematic bias — mean residual ≈ {stats['mean_residual']:.4f} (near zero)."
        )

    # Accuracy bands
    interpretations.append(
        f"📊 {stats['within_10pct']}% of predictions are within 10% of actual value. "
        f"{stats['within_20pct']}% are within 20%."
    )

    if stats["within_10pct"] > 80:
        interpretations.append("✅ Strong predictive accuracy — 80%+ predictions within 10% error.")
    elif stats["within_10pct"] > 60:
        interpretations.append("🟡 Moderate accuracy — consider feature engineering to improve tight-error performance.")
    else:
        interpretations.append("🔴 High error rate — model struggles with precision. Check for outliers in target or missing features.")

    # Heteroscedasticity check (residuals growing with prediction magnitude)
    try:
        corr = float(np.corrcoef(np.abs(residuals), np.abs(y_pred))[0, 1])
        if corr > 0.3:
            patterns.append("heteroscedasticity")
            interpretations.append(
                f"⚠️ Heteroscedasticity detected (|residual| vs |prediction| correlation = {corr:.2f}) — "
                "errors grow larger for bigger predicted values. Try log-transforming the target."
            )
    except Exception:
        pass

    # Data for plots
    plot_data = {
        "y_true":     np.array(y_true).tolist()[:500],
        "y_pred":     np.array(y_pred).tolist()[:500],
        "residuals":  residuals.tolist()[:500],
        "abs_pct":    pct_resid.tolist()[:500],
    }

    return {
        "stats":           stats,
        "patterns":        patterns,
        "interpretations": interpretations,
        "plot_data":       plot_data,
        "model_name":      model_name,
    }


# ─────────────────────────────────────────────────────────────────
# 3. CONFUSION MATRIX  (classification only)
# ─────────────────────────────────────────────────────────────────

def compute_confusion_analysis(
    y_true: pd.Series,
    y_pred: np.ndarray,
    label_encoder,
    model_name: str,
) -> dict[str, Any]:
    """
    Computes confusion matrix + per-class precision/recall
    + plain English interpretation.
    """
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        precision_score, recall_score,
    )

    # Decode labels if needed
    if label_encoder is not None:
        y_true_dec = label_encoder.inverse_transform(np.array(y_true).astype(int))
        y_pred_dec = label_encoder.inverse_transform(np.array(y_pred).astype(int))
    else:
        y_true_dec = np.array(y_true)
        y_pred_dec = np.array(y_pred)

    classes = sorted(list(set(y_true_dec)))
    cm      = confusion_matrix(y_true_dec, y_pred_dec, labels=classes)

    # Per-class metrics
    avg  = "binary" if len(classes) == 2 else "macro"
    prec = precision_score(y_true_dec, y_pred_dec, average=None, labels=classes, zero_division=0)
    rec  = recall_score(y_true_dec, y_pred_dec, average=None, labels=classes, zero_division=0)

    per_class = []
    for i, cls in enumerate(classes):
        per_class.append({
            "class":     str(cls),
            "precision": round(float(prec[i]), 4),
            "recall":    round(float(rec[i]),  4),
            "f1":        round(2 * prec[i] * rec[i] / (prec[i] + rec[i] + 1e-8), 4),
            "support":   int((y_true_dec == cls).sum()),
        })

    # Plain English interpretation
    interpretations: list[str] = []
    overall_acc = round(float((y_true_dec == y_pred_dec).mean()), 4)
    interpretations.append(
        f"Overall accuracy: {overall_acc*100:.1f}% — model correctly classifies "
        f"{int(overall_acc * len(y_true_dec))} of {len(y_true_dec)} test samples."
    )

    # Worst class
    worst = min(per_class, key=lambda x: x["recall"])
    best  = max(per_class, key=lambda x: x["recall"])
    interpretations.append(
        f"Best class: '{best['class']}' (recall={best['recall']:.3f}) — "
        f"model identifies these correctly {best['recall']*100:.1f}% of the time."
    )
    interpretations.append(
        f"Worst class: '{worst['class']}' (recall={worst['recall']:.3f}) — "
        f"model misses {(1-worst['recall'])*100:.1f}% of these cases. "
        + ("Consider SMOTE or class_weight='balanced' if this is an important class." if worst["recall"] < 0.5 else "")
    )

    # False negatives for binary
    if len(classes) == 2:
        fn_count = int(cm[1][0]) if cm.shape == (2, 2) else 0
        fp_count = int(cm[0][1]) if cm.shape == (2, 2) else 0
        interpretations.append(
            f"False Negatives: {fn_count} (missed positive cases) | "
            f"False Positives: {fp_count} (wrongly predicted positive). "
            + ("High FN rate — tune classification threshold if false negatives are costly." if fn_count > fp_count * 2 else "")
        )

    return {
        "confusion_matrix": cm.tolist(),
        "classes":          [str(c) for c in classes],
        "per_class":        per_class,
        "interpretations":  interpretations,
        "overall_accuracy": overall_acc,
        "model_name":       model_name,
    }


# ─────────────────────────────────────────────────────────────────
# 4. TRAIN/TEST DRIFT DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_drift(
    df_train: pd.DataFrame,
    df_test:  pd.DataFrame,
    target_column: str | None = None,
) -> dict[str, Any]:
    """
    Compares feature distributions between train and test sets.
    Uses KS test for numeric, chi-squared for categorical.
    Returns per-feature drift scores + overall drift verdict.
    """
    from scipy import stats as scipy_stats

    drop_cols = [c for c in [target_column] if c and c in df_train.columns]
    train = df_train.drop(columns=drop_cols, errors="ignore")
    test  = df_test.drop(columns=drop_cols, errors="ignore")

    # Align columns
    common_cols = [c for c in train.columns if c in test.columns]
    train = train[common_cols]
    test  = test[common_cols]

    numeric_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols     = train.select_dtypes(exclude=[np.number]).columns.tolist()

    drift_results: list[dict] = []
    high_drift_cols: list[str] = []

    # KS test for numeric
    for col in numeric_cols:
        try:
            ks_stat, p_value = scipy_stats.ks_2samp(
                train[col].dropna(), test[col].dropna()
            )
            drifted = p_value < 0.05

            # Mean shift
            train_mean = float(train[col].mean())
            test_mean  = float(test[col].mean())
            mean_shift_pct = abs(test_mean - train_mean) / (abs(train_mean) + 1e-8) * 100

            if drifted:
                high_drift_cols.append(col)

            drift_results.append({
                "feature":       col,
                "type":          "numeric",
                "ks_statistic":  round(float(ks_stat), 4),
                "p_value":       round(float(p_value), 4),
                "drift_detected": drifted,
                "train_mean":    round(train_mean, 4),
                "test_mean":     round(test_mean, 4),
                "mean_shift_pct": round(mean_shift_pct, 1),
                "severity":      (
                    "🔴 HIGH" if ks_stat > 0.3 else
                    "🟡 MEDIUM" if ks_stat > 0.15 else
                    "🟢 LOW"
                ),
            })
        except Exception:
            continue

    # Chi-squared for categorical
    for col in cat_cols[:10]:  # limit to avoid very large categoricals
        try:
            train_counts = train[col].value_counts()
            test_counts  = test[col].value_counts()
            common_cats  = list(set(train_counts.index) & set(test_counts.index))
            if len(common_cats) < 2:
                continue

            t_vals = np.array([train_counts.get(c, 0) for c in common_cats], dtype=float)
            s_vals = np.array([test_counts.get(c, 0) for c in common_cats], dtype=float)
            t_vals = t_vals / t_vals.sum() * len(test)
            t_vals = np.maximum(t_vals, 1e-8)

            chi2, p_value = scipy_stats.chisquare(s_vals, f_exp=t_vals)
            drifted = p_value < 0.05
            if drifted:
                high_drift_cols.append(col)

            drift_results.append({
                "feature":       col,
                "type":          "categorical",
                "chi2":          round(float(chi2), 4),
                "p_value":       round(float(p_value), 4),
                "drift_detected": drifted,
                "train_categories": int(train[col].nunique()),
                "test_categories":  int(test[col].nunique()),
                "severity":      "🔴 HIGH" if p_value < 0.001 else ("🟡 MEDIUM" if p_value < 0.05 else "🟢 LOW"),
            })
        except Exception:
            continue

    drift_results.sort(key=lambda x: x.get("ks_statistic", x.get("chi2", 0)), reverse=True)

    n_drifted = len(high_drift_cols)
    n_total   = len(drift_results)
    drift_pct = round(n_drifted / max(n_total, 1) * 100, 1)

    if drift_pct > 30:
        verdict = "🔴 HIGH DRIFT — significant distribution shift between train and test. Model may underperform in production."
    elif drift_pct > 10:
        verdict = "🟡 MODERATE DRIFT — some features differ between sets. Monitor model performance closely."
    else:
        verdict = "🟢 LOW DRIFT — train and test distributions are similar. Model should generalize well."

    interpretations = [
        f"{n_drifted} of {n_total} features show statistically significant drift (p < 0.05).",
        verdict,
    ]
    if high_drift_cols:
        interpretations.append(
            f"Highest drift features: {', '.join(high_drift_cols[:5])}. "
            "These features may behave differently in production than in training."
        )

    return {
        "drift_results":    drift_results,
        "high_drift_cols":  high_drift_cols,
        "drift_pct":        drift_pct,
        "verdict":          verdict,
        "interpretations":  interpretations,
        "n_train":          len(df_train),
        "n_test":           len(df_test),
        "n_features_checked": n_total,
    }


# ─────────────────────────────────────────────────────────────────
# 5. MODEL CARD GENERATION
# ─────────────────────────────────────────────────────────────────

def generate_model_card(
    model_name:         str,
    task_type:          str,
    target_column:      str,
    profile:            dict,
    health:             dict,
    leaderboard:        list[dict],
    feature_importance: dict,
    leakage:            dict,
    ts_info:            dict,
    cleaning_report:    dict,
    api_key:            str | None = None,
) -> str:
    """
    Generates a structured model card as a Markdown string.
    """
    best = leaderboard[0] if leaderboard else {}
    now  = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")

    if task_type == "regression":
        metrics_section = f"""
| Metric | Value |
|--------|-------|
| CV R² | {best.get('cv_score', '—')} |
| Train R² | {best.get('train_score', '—')} |
| Test R² | {best.get('test_r2', '—')} |
| MAE | {best.get('mae', '—')} |
| RMSE | {best.get('rmse', '—')} |
"""
    else:
        metrics_section = f"""
| Metric | Value |
|--------|-------|
| CV Accuracy | {best.get('cv_score', '—')} |
| Train Accuracy | {best.get('train_score', '—')} |
| Test Accuracy | {best.get('test_accuracy', '—')} |
| F1 Score | {best.get('f1_score', '—')} |
| ROC-AUC | {best.get('roc_auc', '—')} |
"""

    top_features = "\n".join(
        f"- `{k}`: importance={v:.5f}"
        for k, v in list(feature_importance.items())[:8]
    )

    leakage_note = (
        f"⚠️ **Leakage risk detected**: {leakage.get('leakage_candidates', [])} — verify before deploying."
        if leakage.get("leakage_candidates")
        else "✅ No leakage features detected."
    )

    ts_note = (
        f"⏱️ **Time-series dataset** — column `{ts_info['datetime_column']}`, "
        f"frequency: {ts_info['frequency_guess']}. Chronological split used."
        if ts_info.get("is_timeseries")
        else "Standard random train/test split."
    )

    cleaning_actions = "\n".join(
        f"- {a}" for a in cleaning_report.get("actions", ["No cleaning required."])
    )
    orig  = cleaning_report.get("original_shape", ("?", "?"))
    final = cleaning_report.get("final_shape",    ("?", "?"))

    card = f"""# 🧠 Model Card — AutoML Debugger

**Generated:** {now}
**Model:** {model_name}
**Task:** {task_type.capitalize()}
**Target Column:** `{target_column}`
**Health Score:** {health.get('total', '?')}/100 — {health.get('verdict', '')}

---

## 📊 Dataset

| Property | Value |
|----------|-------|
| Rows | {profile.get('rows', '?'):,} |
| Columns | {profile.get('columns', '?')} |
| Numeric Features | {profile.get('numeric_features', '?')} |
| Categorical Features | {profile.get('categorical_features', '?')} |
| Missing Values | {profile.get('missing_pct', '?')}% |
| Duplicate Rows | {profile.get('duplicate_rows', '?')} |

**Time-series:** {ts_note}

**Data Leakage:** {leakage_note}

---

## 🧹 Preprocessing Applied

Original shape: {orig[0]} × {orig[1]} → Cleaned shape: {final[0]} × {final[1]}

{cleaning_actions}

---

## 🏆 Model Performance
{metrics_section}
**Train/CV Gap:** {abs((best.get('train_score') or 0) - (best.get('cv_score') or 0)):.4f}
{"⚠️ Possible overfitting — gap > 0.1" if abs((best.get('train_score') or 0) - (best.get('cv_score') or 0)) > 0.1 else "✅ Good generalization"}

---

## 🌲 Top Feature Importances

{top_features if top_features else "Feature importance not available."}

---

## ⭐ Health Score Breakdown

| Dimension | Score | Max | Notes |
|-----------|-------|-----|-------|
{"".join(f"| {k} | {v['score']} | {v['max']} | {v['reason']} |" + chr(10) for k, v in health.get('dimensions', {}).items())}
| **TOTAL** | **{health.get('total', '?')}** | **100** | {health.get('verdict', '')} |

---

## ⚠️ Known Limitations

- Model trained on a static snapshot of data — performance may degrade if data distribution changes over time.
- Cross-validated scores reflect performance on training distribution only.
- Always validate on a truly held-out dataset before production deployment.
{"- Leakage features detected — model results may be overly optimistic until these are verified." if leakage.get("leakage_candidates") else ""}

---

## 📋 Intended Use

This model was generated by AutoML Debugger for exploratory and diagnostic purposes.
**Do not deploy to production** without:
1. Verifying feature leakage
2. Validating on a time-held-out test set
3. Human review of model card and feature importances

---

*AutoML Debugger v3.0 · Groq LLaMA 3.3 70B · XGBoost · LightGBM · scikit-learn*
"""
    return card


# ─────────────────────────────────────────────────────────────────
# 6. MODEL SAVE / LOAD
# ─────────────────────────────────────────────────────────────────

def save_model_to_bytes(
    best_pipeline,
    label_encoder,
    numeric_features: list[str],
    categorical_features: list[str],
    task_type: str,
    model_name: str,
    target_column: str,
) -> bytes:
    """Serialise the trained pipeline + metadata to bytes via joblib."""
    import joblib

    bundle = {
        "pipeline":            best_pipeline,
        "label_encoder":       label_encoder,
        "numeric_features":    numeric_features,
        "categorical_features": categorical_features,
        "task_type":           task_type,
        "model_name":          model_name,
        "target_column":       target_column,
        "saved_at":            __import__("datetime").datetime.now().isoformat(),
        "version":             "3.0",
    }

    buf = io.BytesIO()
    joblib.dump(bundle, buf)
    return buf.getvalue()


def load_model_from_bytes(model_bytes: bytes) -> dict[str, Any]:
    """Load a saved model bundle from bytes."""
    import joblib
    buf = io.BytesIO(model_bytes)
    return joblib.load(buf)
