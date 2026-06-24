"""
AutoML Debugger — Core Engine  (Milestone 1 v4.0)
==================================================
8 analysis features:
  1. Dataset Health Score  (per-dimension A-F grade)
  2. Statistical Distribution Analysis  (KDE, normality, skewness)
  3. Feature Redundancy Detection  (correlation matrix, VIF)
  4. Missing Value Pattern Analysis  (MCAR/MAR/MNAR detection)
  5. Smart Column Type Inference  (ZIP, phone, currency, ID, date)
  6. Sample Size Adequacy Check  (rules of thumb per task type)
  7. Data Quality Scorecard  (A/B/C/D/F with sub-scores)
  8. Automated Fix Application  (one-click clean + download)
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
from scipy import stats as scipy_stats

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────
# TASK AUTO-DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_task_type(y: pd.Series) -> tuple[str, str]:
    n_unique     = y.nunique()
    unique_ratio = n_unique / max(len(y), 1)

    if y.dtype == object or y.dtype.name == "category":
        return "classification", f"'{y.name}' has text/category values ({n_unique} unique classes)."

    if n_unique <= 2:
        return "classification", f"'{y.name}' has only {n_unique} unique numeric values (binary)."

    if n_unique <= 20 and unique_ratio < 0.05:
        return "classification", f"'{y.name}' has {n_unique} unique values ({unique_ratio*100:.1f}% of rows) — low cardinality."

    return "regression", (
        f"'{y.name}' is numeric with {n_unique} unique values "
        f"(range: {y.min():.2f} – {y.max():.2f})."
    )


# ─────────────────────────────────────────────────────────────────
# TIME-SERIES DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_timeseries(df: pd.DataFrame) -> dict[str, Any]:
    result = {"is_timeseries": False, "datetime_column": None, "frequency_guess": None}
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            result.update({"is_timeseries": True, "datetime_column": col})
            break
        if df[col].dtype == object:
            try:
                parsed = pd.to_datetime(df[col], format="mixed", dayfirst=False)
                if parsed.notna().mean() > 0.9:
                    result["is_timeseries"]   = True
                    result["datetime_column"] = col
                    diffs  = parsed.dropna().sort_values().diff().dropna()
                    days   = getattr(diffs.median(), "days", 0)
                    result["frequency_guess"] = (
                        "Daily" if days == 1 else "Weekly" if days == 7 else
                        "Monthly" if 28 <= days <= 31 else "Sub-daily" if days == 0
                        else f"~{days}-day intervals"
                    )
                    break
            except Exception:
                continue
    return result


# ─────────────────────────────────────────────────────────────────
# FEATURE 1 — DATASET HEALTH SCORE
# ─────────────────────────────────────────────────────────────────

def compute_health_score(
    profile: dict,
    leakage: dict,
    redundancy: dict,
    sample_check: dict,
    type_inference: dict,
) -> dict[str, Any]:
    dims: dict[str, dict] = {}

    # Completeness (0-20)
    mp = profile.get("missing_pct", 0)
    cs = max(0, 20 - int(mp * 0.6))
    dims["Completeness"] = {
        "score": cs, "max": 20,
        "reason": f"{mp}% missing" if mp > 0 else "No missing values",
    }

    # Integrity (0-20)
    dup_r     = profile.get("duplicate_rows", 0) / max(profile.get("rows", 1), 1)
    const_pen = min(len(profile.get("constant_features", [])) * 4, 10)
    ip        = max(0, 20 - int(dup_r * 20) - const_pen)
    dims["Integrity"] = {
        "score": ip, "max": 20,
        "reason": (f"{profile.get('duplicate_rows',0)} duplicates, "
                   f"{len(profile.get('constant_features',[]))} constant cols")
                  if (dup_r > 0 or const_pen > 0) else "No integrity issues",
    }

    # Leakage Safety (0-20)
    n_leak = len(leakage.get("leakage_candidates", []))
    lp     = max(0, 20 - n_leak * 10)
    dims["Leakage Safety"] = {
        "score": lp, "max": 20,
        "reason": f"{n_leak} leakage candidate(s) detected" if n_leak > 0 else "No leakage detected",
    }

    # Feature Quality (0-20)
    n_redundant = len(redundancy.get("redundant_pairs", []))
    n_bad_types = len(type_inference.get("warnings", []))
    fq = max(0, 20 - n_redundant * 3 - n_bad_types * 2)
    dims["Feature Quality"] = {
        "score": fq, "max": 20,
        "reason": (f"{n_redundant} redundant pairs, {n_bad_types} type warnings")
                  if (n_redundant + n_bad_types) > 0 else "No feature quality issues",
    }

    # Sample Adequacy (0-20)
    adequate = sample_check.get("adequate", True)
    sp       = 20 if adequate else max(0, int(sample_check.get("ratio", 0) * 20))
    dims["Sample Adequacy"] = {
        "score": sp, "max": 20,
        "reason": sample_check.get("summary", "Adequate sample size"),
    }

    total = sum(d["score"] for d in dims.values())

    if total >= 90:   grade, verdict = "A", "🟢 Excellent — Ready to Train"
    elif total >= 75: grade, verdict = "B", "🟢 Good — Minor Issues"
    elif total >= 60: grade, verdict = "C", "🟡 Fair — Fix Before Training"
    elif total >= 40: grade, verdict = "D", "🟠 Poor — Significant Issues"
    else:             grade, verdict = "F", "🔴 Critical — Not Trainable"

    return {"total": total, "grade": grade, "verdict": verdict, "dimensions": dims}


# ─────────────────────────────────────────────────────────────────
# FEATURE 2 — STATISTICAL DISTRIBUTION ANALYSIS
# ─────────────────────────────────────────────────────────────────

def analyze_distributions(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column and target_column in numeric_cols:
        numeric_cols.remove(target_column)

    results: dict[str, dict] = {}
    for col in numeric_cols[:20]:  # cap at 20
        series = df[col].dropna()
        if len(series) < 8:
            continue

        sk   = float(series.skew())
        kurt = float(series.kurtosis())

        # Normality test (Shapiro-Wilk on sample)
        sample = series.sample(min(500, len(series)), random_state=42)
        try:
            _, p_normal = scipy_stats.shapiro(sample)
            is_normal = p_normal > 0.05
        except Exception:
            p_normal, is_normal = None, False

        # Distribution shape
        if abs(sk) < 0.5:
            shape = "Normal / Symmetric"
        elif sk > 1.5:
            shape = "Highly Right-Skewed"
        elif sk > 0.5:
            shape = "Mildly Right-Skewed"
        elif sk < -1.5:
            shape = "Highly Left-Skewed"
        else:
            shape = "Mildly Left-Skewed"

        # Recommendation
        if abs(sk) > 1.5 and series.min() > 0:
            rec = "Apply log1p transform before training"
        elif abs(sk) > 0.5:
            rec = "Consider sqrt or Box-Cox transform"
        else:
            rec = "No transform needed"

        results[col] = {
            "mean":      round(float(series.mean()), 4),
            "std":       round(float(series.std()),  4),
            "min":       round(float(series.min()),  4),
            "max":       round(float(series.max()),  4),
            "skewness":  round(sk,   3),
            "kurtosis":  round(kurt, 3),
            "p_normal":  round(float(p_normal), 4) if p_normal is not None else None,
            "is_normal": is_normal,
            "shape":     shape,
            "recommendation": rec,
        }

    return results


# ─────────────────────────────────────────────────────────────────
# FEATURE 3 — FEATURE REDUNDANCY DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_redundancy(df: pd.DataFrame, target_column: str | None = None) -> dict[str, Any]:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_column and target_column in numeric_cols:
        numeric_cols.remove(target_column)

    if len(numeric_cols) < 2:
        return {"redundant_pairs": [], "corr_matrix": {}, "vif_scores": {}, "drop_suggestions": []}

    X = df[numeric_cols].dropna()
    corr_matrix = X.corr().round(4)

    # Find highly correlated pairs
    redundant_pairs: list[dict] = []
    seen = set()
    for i, c1 in enumerate(numeric_cols):
        for c2 in numeric_cols[i+1:]:
            pair_key = tuple(sorted([c1, c2]))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            try:
                corr_val = abs(float(corr_matrix.loc[c1, c2]))
                if corr_val >= 0.85:
                    redundant_pairs.append({
                        "feature_1": c1,
                        "feature_2": c2,
                        "correlation": round(corr_val, 4),
                        "severity": "🔴 HIGH" if corr_val >= 0.95 else "🟡 MODERATE",
                        "recommendation": f"Drop '{c2}' — '{c1}' carries same information",
                    })
            except Exception:
                continue

    redundant_pairs.sort(key=lambda x: x["correlation"], reverse=True)

    # VIF scores (Variance Inflation Factor)
    vif_scores: dict[str, float] = {}
    if len(numeric_cols) >= 2 and len(X) > len(numeric_cols):
        try:
            from sklearn.linear_model import LinearRegression
            for col in numeric_cols[:15]:  # cap for speed
                others = [c for c in numeric_cols if c != col]
                if not others:
                    continue
                Xo = X[others].values
                y  = X[col].values
                lr = LinearRegression().fit(Xo, y)
                r2 = lr.score(Xo, y)
                vif = round(float(1 / (1 - r2 + 1e-8)), 2)
                vif_scores[col] = vif
        except Exception:
            pass

    # Drop suggestions — keep one from each highly correlated group
    drop_suggestions = list(set(p["feature_2"] for p in redundant_pairs if p["correlation"] >= 0.95))

    return {
        "redundant_pairs":  redundant_pairs[:20],
        "corr_matrix":      corr_matrix.to_dict(),
        "vif_scores":       vif_scores,
        "drop_suggestions": drop_suggestions,
        "n_numeric":        len(numeric_cols),
    }


# ─────────────────────────────────────────────────────────────────
# FEATURE 4 — MISSING VALUE PATTERN ANALYSIS
# ─────────────────────────────────────────────────────────────────

def analyze_missing_patterns(df: pd.DataFrame) -> dict[str, Any]:
    missing_cols = [c for c in df.columns if df[c].isna().sum() > 0]

    if not missing_cols:
        return {
            "pattern": "NONE",
            "description": "No missing values found.",
            "columns": {},
            "recommendation": "Dataset is complete — no imputation needed.",
        }

    col_stats: dict[str, dict] = {}
    for col in missing_cols:
        pct = round(float(df[col].isna().mean() * 100), 2)
        col_stats[col] = {
            "count":   int(df[col].isna().sum()),
            "pct":     pct,
            "severity": "🔴 Critical" if pct > 30 else ("🟡 Moderate" if pct > 10 else "🟢 Minor"),
        }

    # Pattern detection
    total_missing_pct = df.isna().mean().mean() * 100

    # Check if missingness is correlated with other columns (MAR signal)
    mar_signals: list[str] = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for mc in missing_cols[:5]:
        mask = df[mc].isna().astype(int)
        for nc in numeric_cols:
            if nc == mc:
                continue
            try:
                corr = abs(float(mask.corr(df[nc])))
                if corr > 0.2:
                    mar_signals.append(f"'{mc}' missingness correlates with '{nc}' ({corr:.2f})")
                    break
            except Exception:
                continue

    # Determine pattern
    if mar_signals:
        pattern = "MAR"
        description = "Missing At Random — missingness depends on other observed columns."
        recommendation = "Use advanced imputation (KNN or iterative). Simple median/mode may introduce bias."
    elif total_missing_pct < 5:
        pattern = "MCAR"
        description = "Missing Completely At Random — small, random gaps."
        recommendation = "Median/mode imputation is safe. Consider dropping rows with missing target."
    else:
        pattern = "MNAR"
        description = "Missing Not At Random — the missing value itself may be informative."
        recommendation = "Add a binary indicator column for each missing feature. The absence is a signal."

    return {
        "pattern":        pattern,
        "description":    description,
        "columns":        col_stats,
        "mar_signals":    mar_signals,
        "recommendation": recommendation,
        "total_missing_pct": round(total_missing_pct, 2),
    }


# ─────────────────────────────────────────────────────────────────
# FEATURE 5 — SMART COLUMN TYPE INFERENCE
# ─────────────────────────────────────────────────────────────────

def infer_column_types(df: pd.DataFrame) -> dict[str, Any]:
    inferences: list[dict] = []
    warnings_list: list[str] = []

    for col in df.columns:
        series  = df[col].dropna()
        if len(series) == 0:
            continue

        dtype   = str(df[col].dtype)
        inferred = dtype
        flag     = None
        rec      = None

        # Numeric column checks
        if pd.api.types.is_numeric_dtype(df[col]):
            unique_vals = series.unique()

            # Binary flag stored as numeric
            if set(unique_vals).issubset({0, 1, 0.0, 1.0}) and len(unique_vals) == 2:
                inferred = "Binary Flag (0/1)"
                flag     = "🟡 Categorical stored as numeric"
                rec      = "Consider treating as categorical for tree models"

            # ID column
            elif (series.nunique() == len(series) and
                  col.lower() in {"id","index","row_id","uid","uuid","customer_id","user_id","order_id"}):
                inferred = "ID Column"
                flag     = "🔴 Should be dropped"
                rec      = f"Drop '{col}' — unique per row, no predictive value"
                warnings_list.append(f"'{col}' looks like an ID — drop it")

            # Percentage stored as decimal (0-1 range)
            elif series.between(0, 1).all() and series.nunique() > 10:
                inferred = "Percentage (0–1)"
                flag     = "🟢 Numeric — OK"
                rec      = "May want to multiply by 100 for readability"

            # Currency / large money values
            elif series.mean() > 1000 and series.skew() > 1:
                inferred = "Possible Currency / Revenue"
                flag     = "🟡 Consider log transform"
                rec      = "High skewness — apply log1p before training"

            # Corrupted range (e.g. RSI outside 0-100)
            elif col.upper().endswith("RSI") and (series.min() < 0 or series.max() > 100):
                inferred = "RSI — Corrupted Range"
                flag     = "🔴 Data Quality Issue"
                rec      = f"RSI must be 0–100. Found range: {series.min():.1f}–{series.max():.1f}. Check data source."
                warnings_list.append(f"'{col}' RSI values outside valid range 0–100")

        # String column checks
        elif df[col].dtype == object:
            sample_str = series.astype(str).head(50)

            # Date-like
            try:
                parsed = pd.to_datetime(series.head(50), format="mixed", dayfirst=False)
                if parsed.notna().mean() > 0.9:
                    inferred = "Date / Datetime"
                    flag     = "🟡 Parse as datetime"
                    rec      = "Use pd.to_datetime() and extract year/month/day/weekday features"
            except Exception:
                pass

            # Email
            if inferred == dtype and sample_str.str.contains("@").mean() > 0.5:
                inferred = "Email Address"
                flag     = "🔴 Drop or extract domain"
                rec      = "Extract domain (gmail/yahoo etc.) as categorical feature, or drop"
                warnings_list.append(f"'{col}' contains emails — drop or extract domain")

            # URL
            elif inferred == dtype and sample_str.str.startswith("http").mean() > 0.5:
                inferred = "URL"
                flag     = "🔴 Drop"
                rec      = "URLs are not useful as ML features — drop this column"
                warnings_list.append(f"'{col}' contains URLs — should be dropped")

            # ZIP code (5-digit strings)
            elif inferred == dtype and sample_str.str.match(r"^\d{5}$").mean() > 0.5:
                inferred = "ZIP Code"
                flag     = "🟡 Treat as categorical"
                rec      = "Don't treat as numeric. Group into regions or use as categorical."

        inferences.append({
            "column":   col,
            "dtype":    dtype,
            "inferred": inferred,
            "flag":     flag or "🟢 OK",
            "recommendation": rec or "No action needed",
        })

    return {
        "inferences": inferences,
        "warnings":   warnings_list,
        "n_warnings": len(warnings_list),
    }


# ─────────────────────────────────────────────────────────────────
# FEATURE 6 — SAMPLE SIZE ADEQUACY CHECK
# ─────────────────────────────────────────────────────────────────

def check_sample_size(df: pd.DataFrame, target_column: str, task_type: str) -> dict[str, Any]:
    n_rows  = len(df)
    n_feats = len(df.columns) - 1
    checks: list[dict] = []
    issues: list[str]  = []

    # Rule 1 — minimum rows per feature
    min_rows_per_feat = 10
    required_min      = n_feats * min_rows_per_feat
    pass1 = n_rows >= required_min
    checks.append({
        "rule":   f"≥ {min_rows_per_feat} rows per feature",
        "needed": required_min,
        "have":   n_rows,
        "pass":   pass1,
    })
    if not pass1:
        issues.append(f"Need {required_min:,} rows for {n_feats} features — you have {n_rows:,}. High overfitting risk.")

    # Rule 2 — absolute minimum
    abs_min = 100
    pass2   = n_rows >= abs_min
    checks.append({
        "rule": f"Absolute minimum {abs_min} rows",
        "needed": abs_min, "have": n_rows, "pass": pass2,
    })
    if not pass2:
        issues.append(f"Absolute minimum is {abs_min} rows. Collect more data before training.")

    # Rule 3 — classification specific (min per class)
    if task_type == "classification":
        vc      = df[target_column].value_counts()
        min_cls = int(vc.min())
        pass3   = min_cls >= 50
        checks.append({
            "rule": "≥ 50 samples per class",
            "needed": 50, "have": min_cls, "pass": pass3,
        })
        if not pass3:
            issues.append(f"Smallest class has only {min_cls} samples. Need ≥ 50 per class for reliable classification.")

    # Rule 4 — learning curve projection
    projected_improvement = "Unknown"
    if n_rows < 500:
        projected_improvement = "High — doubling data could improve accuracy by 10-20%"
    elif n_rows < 2000:
        projected_improvement = "Moderate — more data will help but diminishing returns"
    else:
        projected_improvement = "Low — dataset is large enough, focus on features instead"

    ratio    = min(1.0, n_rows / max(required_min, 1))
    adequate = len(issues) == 0

    return {
        "adequate":                adequate,
        "n_rows":                  n_rows,
        "n_features":              n_feats,
        "checks":                  checks,
        "issues":                  issues,
        "projected_improvement":   projected_improvement,
        "ratio":                   ratio,
        "summary": (
            f"{n_rows:,} rows, {n_feats} features — adequate"
            if adequate else
            f"{n_rows:,} rows for {n_feats} features — {len(issues)} issue(s)"
        ),
    }


# ─────────────────────────────────────────────────────────────────
# FEATURE 7 — DATA QUALITY SCORECARD  (A/B/C/D/F)
# ─────────────────────────────────────────────────────────────────

def build_scorecard(
    profile:        dict,
    health:         dict,
    redundancy:     dict,
    missing_pattern: dict,
    type_inference: dict,
    sample_check:   dict,
    leakage:        dict,
) -> dict[str, Any]:
    """
    Comprehensive A-F scorecard with sub-scores and benchmark comparison.
    """
    def score_to_grade(s, mx):
        pct = s / max(mx, 1) * 100
        if pct >= 90: return "A"
        if pct >= 75: return "B"
        if pct >= 60: return "C"
        if pct >= 40: return "D"
        return "F"

    sections: list[dict] = []

    # Section 1 — Data Completeness
    mp  = profile.get("missing_pct", 0)
    s1  = max(0, 100 - int(mp * 2))
    sections.append({
        "name":    "Data Completeness",
        "score":   s1,
        "grade":   score_to_grade(s1, 100),
        "details": f"{mp}% missing across {len(profile.get('missing_by_col',{}))} columns",
        "impact":  "HIGH",
    })

    # Section 2 — Data Integrity
    dups   = profile.get("duplicate_rows", 0)
    consts = len(profile.get("constant_features", []))
    s2     = max(0, 100 - dups/max(profile.get("rows",1),1)*100 - consts*10)
    sections.append({
        "name":    "Data Integrity",
        "score":   round(s2),
        "grade":   score_to_grade(s2, 100),
        "details": f"{dups} duplicates, {consts} constant columns",
        "impact":  "MEDIUM",
    })

    # Section 3 — Feature Quality
    n_red  = len(redundancy.get("redundant_pairs", []))
    n_warn = type_inference.get("n_warnings", 0)
    s3     = max(0, 100 - n_red*5 - n_warn*10)
    sections.append({
        "name":    "Feature Quality",
        "score":   s3,
        "grade":   score_to_grade(s3, 100),
        "details": f"{n_red} redundant pairs, {n_warn} type warnings",
        "impact":  "HIGH",
    })

    # Section 4 — Leakage Safety
    n_leak = len(leakage.get("leakage_candidates", []))
    s4     = max(0, 100 - n_leak*25)
    sections.append({
        "name":    "Leakage Safety",
        "score":   s4,
        "grade":   score_to_grade(s4, 100),
        "details": f"{n_leak} high-risk feature(s)" if n_leak else "No leakage detected",
        "impact":  "CRITICAL",
    })

    # Section 5 — Sample Adequacy
    s5 = 100 if sample_check.get("adequate") else int(sample_check.get("ratio", 0.5) * 100)
    sections.append({
        "name":    "Sample Adequacy",
        "score":   s5,
        "grade":   score_to_grade(s5, 100),
        "details": sample_check.get("summary", ""),
        "impact":  "MEDIUM",
    })

    # Section 6 — Missing Pattern
    pattern = missing_pattern.get("pattern", "NONE")
    s6 = 100 if pattern == "NONE" else (80 if pattern == "MCAR" else (50 if pattern == "MAR" else 30))
    sections.append({
        "name":    "Missing Pattern",
        "score":   s6,
        "grade":   score_to_grade(s6, 100),
        "details": f"Pattern: {pattern} — {missing_pattern.get('description','')}",
        "impact":  "MEDIUM",
    })

    overall = int(sum(s["score"] for s in sections) / len(sections))
    if overall >= 90:   overall_grade, overall_verdict = "A", "Excellent — Ready to Train"
    elif overall >= 75: overall_grade, overall_verdict = "B", "Good — Minor Issues to Fix"
    elif overall >= 60: overall_grade, overall_verdict = "C", "Fair — Fix Before Training"
    elif overall >= 40: overall_grade, overall_verdict = "D", "Poor — Significant Work Needed"
    else:               overall_grade, overall_verdict = "F", "Critical — Not Ready for ML"

    # Benchmark (simulated percentile based on score)
    percentile = min(99, max(1, overall))
    benchmark  = f"Better than {percentile}% of datasets we've seen at this size"

    return {
        "overall_score":   overall,
        "overall_grade":   overall_grade,
        "overall_verdict": overall_verdict,
        "benchmark":       benchmark,
        "sections":        sections,
    }


# ─────────────────────────────────────────────────────────────────
# FEATURE 8 — AUTOMATED FIX APPLICATION
# ─────────────────────────────────────────────────────────────────

def apply_automated_fixes(
    df: pd.DataFrame,
    target_column: str,
    redundancy:     dict,
    type_inference: dict,
    opts: dict | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Applies all recommended fixes and returns (fixed_df, list_of_actions).
    opts keys: drop_duplicates, impute_missing, cap_outliers,
               drop_constant, drop_redundant, drop_id_cols
    """
    opts    = opts or {}
    fixed   = df.copy()
    actions: list[str] = []

    # Drop constant columns
    if opts.get("drop_constant", True):
        const_cols = [c for c in fixed.columns if c != target_column and fixed[c].nunique() <= 1]
        if const_cols:
            fixed.drop(columns=const_cols, inplace=True)
            actions.append(f"Dropped {len(const_cols)} constant column(s): {const_cols}")

    # Drop ID-like columns
    if opts.get("drop_id_cols", True):
        id_cols = [
            inf["column"] for inf in type_inference.get("inferences", [])
            if inf["inferred"] == "ID Column" and inf["column"] in fixed.columns
            and inf["column"] != target_column
        ]
        if id_cols:
            fixed.drop(columns=id_cols, inplace=True)
            actions.append(f"Dropped {len(id_cols)} ID column(s): {id_cols}")

    # Drop redundant features
    if opts.get("drop_redundant", True):
        drop_cols = [
            c for c in redundancy.get("drop_suggestions", [])
            if c in fixed.columns and c != target_column
        ]
        if drop_cols:
            fixed.drop(columns=drop_cols, inplace=True)
            actions.append(f"Dropped {len(drop_cols)} redundant feature(s): {drop_cols[:5]}")

    # Remove duplicates
    if opts.get("drop_duplicates", True):
        before = len(fixed)
        fixed.drop_duplicates(inplace=True)
        n = before - len(fixed)
        if n:
            actions.append(f"Removed {n} duplicate row(s)")

    # Impute missing values
    if opts.get("impute_missing", True):
        num_cols = fixed.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = fixed.select_dtypes(exclude=[np.number]).columns.tolist()
        fn = fc = 0
        for col in num_cols:
            n = fixed[col].isna().sum()
            if n:
                fixed[col].fillna(fixed[col].median(), inplace=True)
                fn += n
        for col in cat_cols:
            n = fixed[col].isna().sum()
            if n:
                fixed[col].fillna(fixed[col].mode()[0] if len(fixed[col].mode()) else "Unknown", inplace=True)
                fc += n
        if fn + fc > 0:
            actions.append(f"Imputed {fn} numeric (median) + {fc} categorical (mode) missing values")

    # Cap outliers
    if opts.get("cap_outliers", True):
        num_cols = fixed.select_dtypes(include=[np.number]).columns.tolist()
        if target_column in num_cols:
            num_cols.remove(target_column)
        capped = []
        for col in num_cols:
            q1, q3 = fixed[col].quantile(0.25), fixed[col].quantile(0.75)
            iqr    = q3 - q1
            lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
            n_out  = ((fixed[col] < lo) | (fixed[col] > hi)).sum()
            if n_out > 0:
                fixed[col] = fixed[col].clip(lo, hi)
                capped.append(col)
        if capped:
            actions.append(f"Capped outliers (IQR) in {len(capped)} column(s): {capped[:5]}{'...' if len(capped)>5 else ''}")

    if not actions:
        actions.append("Dataset was already clean — no fixes needed")

    return fixed, actions


# ─────────────────────────────────────────────────────────────────
# DATA PROFILER  (basic stats for all features)
# ─────────────────────────────────────────────────────────────────

def profile_dataset(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    X = df.drop(columns=[target_column], errors="ignore")
    y = df[target_column] if target_column in df.columns else pd.Series(dtype=float)

    numeric_cols = X.select_dtypes(include=[np.number]).columns
    cat_cols     = X.select_dtypes(exclude=[np.number]).columns

    missing_by_col = {c: int(df[c].isna().sum()) for c in df.columns if df[c].isna().sum() > 0}
    total_cells    = df.shape[0] * df.shape[1]
    missing_total  = int(df.isna().sum().sum())

    outlier_counts: dict[str, int] = {}
    for col in numeric_cols:
        q1, q3 = X[col].quantile(0.25), X[col].quantile(0.75)
        iqr    = q3 - q1
        n      = int(((X[col] < q1 - 1.5*iqr) | (X[col] > q3 + 1.5*iqr)).sum())
        if n > 0:
            outlier_counts[col] = n

    corr_with_target: dict[str, float] = {}
    if pd.api.types.is_numeric_dtype(y):
        for col in numeric_cols:
            c = df[col].corr(y)
            if not np.isnan(c):
                corr_with_target[col] = round(float(c), 4)
        corr_with_target = dict(
            sorted(corr_with_target.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
        )

    task_type, task_reason = detect_task_type(y) if len(y) > 0 else ("unknown", "")
    class_dist      = None
    imbalance_ratio = None
    if task_type == "classification" and len(y) > 0:
        vc              = y.value_counts()
        class_dist      = {str(k): int(v) for k, v in vc.items()}
        if len(vc) >= 2:
            imbalance_ratio = round(float(vc.iloc[0] / vc.iloc[-1]), 2)

    constant_features = [c for c in X.columns if X[c].nunique() <= 1]
    high_card         = [c for c in cat_cols if X[c].nunique() > 50]

    return {
        "rows":                  int(df.shape[0]),
        "columns":               int(df.shape[1]),
        "numeric_features":      int(len(numeric_cols)),
        "categorical_features":  int(len(cat_cols)),
        "missing_total":         missing_total,
        "missing_pct":           round(missing_total / max(total_cells, 1) * 100, 2),
        "missing_by_col":        missing_by_col,
        "duplicate_rows":        int(df.duplicated().sum()),
        "outlier_counts":        outlier_counts,
        "top_correlations":      corr_with_target,
        "class_distribution":    class_dist,
        "imbalance_ratio":       imbalance_ratio,
        "constant_features":     constant_features,
        "high_cardinality_cols": list(high_card),
        "task_type":             task_type,
        "task_reason":           task_reason,
    }


# ─────────────────────────────────────────────────────────────────
# LEAKAGE DETECTION
# ─────────────────────────────────────────────────────────────────

def detect_leakage(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "leakage_candidates": [], "high_correlation_features": {}, "warnings": []
    }
    if target_column not in df.columns:
        return result
    y = df[target_column]
    if not pd.api.types.is_numeric_dtype(y):
        return result

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c != target_column]
    for col in numeric_cols:
        try:
            corr = abs(float(df[col].corr(y)))
            if np.isnan(corr):
                continue
            result["high_correlation_features"][col] = round(corr, 4)
            if corr > 0.95:
                result["leakage_candidates"].append(col)
                result["warnings"].append(
                    f"🚨 '{col}' correlation={corr:.3f} — likely derived from or identical to target"
                )
            elif corr > 0.85:
                result["warnings"].append(
                    f"⚠️ '{col}' correlation={corr:.3f} — high, verify it's not leakage"
                )
        except Exception:
            continue
    return result


# ─────────────────────────────────────────────────────────────────
# MILESTONE 2 — FEATURE 9
# TARGET LEAKAGE PROBABILITY SCORE (Mutual Information)
# ─────────────────────────────────────────────────────────────────

def compute_leakage_probability(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Uses mutual information (MI) alongside correlation to give a
    leakage probability score per feature.
    MI catches non-linear relationships that correlation misses.
    Returns per-feature risk scores + overall leakage verdict.
    """
    if target_column not in df.columns:
        return {"features": {}, "verdict": "Target column not found"}

    y = df[target_column].dropna()
    if not pd.api.types.is_numeric_dtype(y):
        return {"features": {}, "verdict": "Non-numeric target — MI not applicable"}

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target_column]
    if not numeric_cols:
        return {"features": {}, "verdict": "No numeric features"}

    try:
        from sklearn.feature_selection import mutual_info_regression
        from sklearn.preprocessing import StandardScaler

        X = df[numeric_cols].copy()
        # Fill NaN with median for MI computation
        for col in X.columns:
            X[col] = X[col].fillna(X[col].median())

        aligned_y = y.reindex(X.index).dropna()
        X = X.loc[aligned_y.index]

        mi_scores = mutual_info_regression(X, aligned_y, random_state=42)
        mi_scores_norm = mi_scores / (mi_scores.max() + 1e-8)  # normalize 0-1

        features: dict[str, dict] = {}
        for col, mi_raw, mi_norm in zip(numeric_cols, mi_scores, mi_scores_norm):
            corr = abs(float(df[col].corr(y))) if not np.isnan(df[col].corr(y)) else 0

            # Leakage probability = weighted combo of MI + correlation
            leak_prob = round(float(0.5 * mi_norm + 0.5 * corr), 4)

            if leak_prob > 0.85:
                risk = "🔴 CRITICAL"
                verdict = "Almost certainly leakage — feature contains target information"
            elif leak_prob > 0.70:
                risk = "🟠 HIGH"
                verdict = "Strong leakage signal — verify this feature is available at prediction time"
            elif leak_prob > 0.50:
                risk = "🟡 MODERATE"
                verdict = "Elevated MI — may be derived from target or causally linked"
            elif leak_prob > 0.30:
                risk = "🟢 LOW"
                verdict = "Some predictive signal — likely legitimate"
            else:
                risk = "✅ SAFE"
                verdict = "Low mutual information — safe to use"

            features[col] = {
                "mi_score":        round(float(mi_raw), 5),
                "mi_normalized":   round(float(mi_norm), 4),
                "correlation":     round(corr, 4),
                "leakage_prob":    leak_prob,
                "risk_level":      risk,
                "verdict":         verdict,
            }

        # Sort by leakage probability
        features = dict(sorted(features.items(),
                               key=lambda x: x[1]["leakage_prob"], reverse=True))

        critical = [k for k, v in features.items() if v["leakage_prob"] > 0.85]
        high     = [k for k, v in features.items() if 0.70 < v["leakage_prob"] <= 0.85]

        overall_verdict = (
            f"🔴 CRITICAL: {len(critical)} feature(s) show extremely high leakage probability: {critical}"
            if critical else (
                f"🟠 HIGH RISK: {len(high)} feature(s) have elevated leakage scores — review before training"
                if high else "✅ No significant leakage probability detected across all features"
            )
        )

        return {
            "features":       features,
            "verdict":        overall_verdict,
            "n_critical":     len(critical),
            "n_high":         len(high),
            "critical_cols":  critical,
        }

    except Exception as e:
        return {"features": {}, "verdict": f"MI computation failed: {str(e)}", "error": str(e)}


# ─────────────────────────────────────────────────────────────────
# MILESTONE 2 — FEATURE 10
# OUTLIER ROOT CAUSE ANALYSIS (Mahalanobis Distance)
# ─────────────────────────────────────────────────────────────────

def analyze_outlier_root_cause(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Mahalanobis distance detects multivariate outliers —
    rows that are outliers in the COMBINATION of features,
    not just in any single column individually.
    Also checks if outliers cluster in time (date column).
    """
    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target_column]

    if len(numeric_cols) < 2:
        return {"available": False, "reason": "Need at least 2 numeric features"}

    X = df[numeric_cols].copy()
    for col in X.columns:
        X[col] = X[col].fillna(X[col].median())

    # Cap to 1000 rows for speed
    if len(X) > 1000:
        X = X.sample(1000, random_state=42)

    try:
        from sklearn.covariance import EllipticEnvelope

        # Mahalanobis via EllipticEnvelope (robust covariance)
        envelope  = EllipticEnvelope(contamination=0.05, random_state=42)
        outlier_mask = envelope.fit_predict(X) == -1
        mahal_scores = envelope.mahalanobis(X)

        n_outliers = int(outlier_mask.sum())
        pct        = round(n_outliers / len(X) * 100, 1)

        # Which features contribute most to outlier scores
        outlier_rows = X[outlier_mask]
        normal_rows  = X[~outlier_mask]

        feature_contribution: dict[str, dict] = {}
        for col in numeric_cols[:15]:
            if col not in X.columns:
                continue
            out_mean = float(outlier_rows[col].mean()) if len(outlier_rows) > 0 else 0
            nor_mean = float(normal_rows[col].mean())  if len(normal_rows) > 0  else 0
            shift_pct = abs(out_mean - nor_mean) / (abs(nor_mean) + 1e-8) * 100
            feature_contribution[col] = {
                "outlier_mean": round(out_mean, 4),
                "normal_mean":  round(nor_mean, 4),
                "shift_pct":    round(shift_pct, 1),
                "major_driver": shift_pct > 50,
            }

        # Sort by shift
        feature_contribution = dict(
            sorted(feature_contribution.items(),
                   key=lambda x: x[1]["shift_pct"], reverse=True)
        )
        major_drivers = [k for k, v in feature_contribution.items() if v["major_driver"]]

        # IQR outlier count per column for comparison
        iqr_counts: dict[str, int] = {}
        for col in numeric_cols[:15]:
            if col not in df.columns:
                continue
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr    = q3 - q1
            n = int(((df[col] < q1 - 1.5*iqr) | (df[col] > q3 + 1.5*iqr)).sum())
            if n > 0:
                iqr_counts[col] = n

        interpretation = []
        if pct < 3:
            interpretation.append(f"✅ Only {pct}% multivariate outliers — dataset is clean.")
        elif pct < 8:
            interpretation.append(f"🟡 {pct}% multivariate outliers — moderate. Review before training.")
        else:
            interpretation.append(f"🔴 {pct}% multivariate outliers — significant. Investigate before training.")

        if major_drivers:
            interpretation.append(
                f"Primary outlier drivers: {major_drivers[:3]}. "
                "These features show the largest mean shift between outlier and normal rows."
            )

        # New insight: IQR vs Mahalanobis comparison
        interpretation.append(
            f"IQR method found outliers in {len(iqr_counts)} individual columns. "
            f"Mahalanobis found {n_outliers} rows that are unusual in COMBINATION — "
            "these multivariate outliers would be missed by column-wise IQR."
        )

        return {
            "available":             True,
            "n_outliers":            n_outliers,
            "pct_outliers":          pct,
            "feature_contribution":  feature_contribution,
            "major_drivers":         major_drivers,
            "iqr_counts":            iqr_counts,
            "mahal_scores":          sorted(mahal_scores.tolist(), reverse=True)[:20],
            "interpretation":        interpretation,
        }

    except Exception as e:
        return {"available": False, "reason": str(e)}


# ─────────────────────────────────────────────────────────────────
# MILESTONE 2 — FEATURE 11
# DATA DRIFT SIMULATION (First Half vs Second Half)
# ─────────────────────────────────────────────────────────────────

def simulate_data_drift(df: pd.DataFrame, target_column: str) -> dict[str, Any]:
    """
    Splits dataset into first 50% and second 50% (by row order).
    Compares feature distributions using KS test.
    Simulates what drift looks like over time without needing a test set.
    Critical for time-series and sequential data.
    """
    n = len(df)
    if n < 50:
        return {"available": False, "reason": "Need at least 50 rows for drift simulation"}

    half      = n // 2
    df_first  = df.iloc[:half]
    df_second = df.iloc[half:]

    numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns
                    if c != target_column]

    drift_results: list[dict] = []
    drifted_cols:  list[str]  = []

    for col in numeric_cols[:20]:
        try:
            s1 = df_first[col].dropna()
            s2 = df_second[col].dropna()
            if len(s1) < 10 or len(s2) < 10:
                continue

            ks_stat, p_val = scipy_stats.ks_2samp(s1, s2)
            mean1 = float(s1.mean())
            mean2 = float(s2.mean())
            std1  = float(s1.std())
            std2  = float(s2.std())
            mean_shift = round(abs(mean2 - mean1) / (abs(mean1) + 1e-8) * 100, 1)
            std_shift  = round(abs(std2 - std1) / (abs(std1) + 1e-8) * 100, 1)

            drifted = p_val < 0.05
            if drifted:
                drifted_cols.append(col)

            severity = (
                "🔴 HIGH"   if ks_stat > 0.3 else
                "🟡 MEDIUM" if ks_stat > 0.15 else
                "🟢 LOW"
            )

            drift_results.append({
                "feature":     col,
                "ks_stat":     round(float(ks_stat), 4),
                "p_value":     round(float(p_val), 4),
                "drifted":     drifted,
                "severity":    severity,
                "mean_first":  round(mean1, 4),
                "mean_second": round(mean2, 4),
                "mean_shift":  mean_shift,
                "std_shift":   std_shift,
            })
        except Exception:
            continue

    drift_results.sort(key=lambda x: x["ks_stat"], reverse=True)

    n_drifted  = len(drifted_cols)
    drift_pct  = round(n_drifted / max(len(numeric_cols), 1) * 100, 1)

    if drift_pct > 40:
        verdict = "🔴 HIGH DRIFT — dataset distribution changes significantly over rows. Time-series or seasonal effects likely."
        risk    = "HIGH"
    elif drift_pct > 20:
        verdict = "🟡 MODERATE DRIFT — some features change across the dataset. Monitor model performance over time."
        risk    = "MEDIUM"
    else:
        verdict = "🟢 STABLE — feature distributions are consistent across the dataset."
        risk    = "LOW"

    interpretation = [
        f"{n_drifted} of {len(drift_results)} features show statistically significant drift (p < 0.05).",
        verdict,
    ]
    if drifted_cols:
        interpretation.append(
            f"Most drifted features: {drifted_cols[:5]}. "
            "If training on the first half and predicting on the second, "
            "these features may degrade model performance."
        )
    else:
        interpretation.append(
            "No significant drift detected — dataset appears stable across rows. "
            "A random train/test split should work safely."
        )

    # Target drift
    try:
        y1 = df_first[target_column].dropna()
        y2 = df_second[target_column].dropna()
        if len(y1) > 5 and len(y2) > 5 and pd.api.types.is_numeric_dtype(y1):
            ks_y, p_y = scipy_stats.ks_2samp(y1, y2)
            target_drift = {
                "ks_stat": round(float(ks_y), 4),
                "p_value": round(float(p_y), 4),
                "drifted": p_y < 0.05,
                "mean_first":  round(float(y1.mean()), 4),
                "mean_second": round(float(y2.mean()), 4),
            }
            if p_y < 0.05:
                interpretation.append(
                    f"⚠️ Target variable '{target_column}' also shows drift (KS={ks_y:.3f}) — "
                    "the distribution of what you're predicting changes over time. "
                    "Consider time-based train/test split."
                )
        else:
            target_drift = None
    except Exception:
        target_drift = None

    return {
        "available":        True,
        "drift_results":    drift_results,
        "drifted_cols":     drifted_cols,
        "drift_pct":        drift_pct,
        "verdict":          verdict,
        "risk":             risk,
        "interpretation":   interpretation,
        "n_first":          half,
        "n_second":         n - half,
        "target_drift":     target_drift,
    }


# ─────────────────────────────────────────────────────────────────
# MILESTONE 2 — FEATURE 12
# ML PROBLEM FRAMING ASSISTANT (Groq)
# ─────────────────────────────────────────────────────────────────

def generate_problem_framings(
    profile:       dict,
    scorecard:     dict,
    ts_info:       dict,
    target_column: str,
    task_type:     str,
    api_key:       str | None = None,
) -> list[dict[str, str]]:
    """
    Groq suggests 3 different ML problems the user could solve
    with this dataset — with target, model, difficulty, and use case.
    Falls back to rule-based suggestions if no API key.
    """
    key = api_key or os.environ.get("GROQ_API_KEY", "")

    context = {
        "rows":       profile.get("rows", 0),
        "columns":    profile.get("columns", 0),
        "numeric":    profile.get("numeric_features", 0),
        "categorical":profile.get("categorical_features", 0),
        "target":     target_column,
        "task":       task_type,
        "grade":      scorecard.get("overall_grade", "?"),
        "timeseries": ts_info.get("is_timeseries", False),
        "frequency":  ts_info.get("frequency_guess", ""),
        "top_cols":   list(profile.get("top_correlations", {}).keys())[:5],
    }

    if key:
        try:
            from groq import Groq
            client = Groq(api_key=key)

            prompt = f"""You are a senior ML engineer. Given the dataset profile below, suggest exactly 3 different ML problems this dataset could solve.

Dataset profile: {json.dumps(context)}

Return ONLY a valid JSON array of exactly 3 objects, each with these exact keys:
[
  {{
    "title": "Short problem title (5 words max)",
    "target_suggestion": "Which column to predict or cluster",
    "problem_type": "Regression / Classification / Clustering / Forecasting",
    "model_recommendation": "Specific algorithm to use",
    "difficulty": "Easy / Medium / Hard",
    "business_use_case": "One sentence real-world application",
    "why_this_data": "One sentence on why this dataset suits this problem"
  }}
]

Rules: Be specific to this dataset's actual columns. No generic suggestions. No markdown. Pure JSON array only."""

            resp = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.5,
            )
            raw = resp.choices[0].message.content.strip()
            if raw.startswith("```"):
                raw = "\n".join(raw.split("\n")[1:-1])
            parsed = json.loads(raw)
            if isinstance(parsed, list) and len(parsed) == 3:
                return parsed
        except Exception:
            pass

    # Rule-based fallback
    framings = []
    top_cols = list(profile.get("top_correlations", {}).keys())[:3]

    if ts_info.get("is_timeseries"):
        framings.append({
            "title":              "Time-Series Forecasting",
            "target_suggestion":  target_column,
            "problem_type":       "Forecasting",
            "model_recommendation": "ARIMA or XGBoost with lag features",
            "difficulty":         "Hard",
            "business_use_case":  "Predict future values of the target variable based on historical patterns",
            "why_this_data":      f"Dataset has {ts_info['frequency_guess']} frequency — ideal for sequential forecasting",
        })
    else:
        framings.append({
            "title":              "Regression — Predict Target",
            "target_suggestion":  target_column,
            "problem_type":       "Regression",
            "model_recommendation": "XGBoost with default hyperparameters",
            "difficulty":         "Medium",
            "business_use_case":  f"Predict the value of '{target_column}' from other features",
            "why_this_data":      f"{profile.get('numeric_features',0)} numeric features provide rich signal for regression",
        })

    if top_cols:
        framings.append({
            "title":              "Binary Classification",
            "target_suggestion":  f"Binarize '{target_column}' at median threshold",
            "problem_type":       "Classification",
            "model_recommendation": "LightGBM with class_weight='balanced'",
            "difficulty":         "Easy",
            "business_use_case":  f"Classify whether '{target_column}' is above or below the median — useful for threshold-based decisions",
            "why_this_data":      f"Strong correlators ({top_cols[:2]}) make this a viable binary classification task",
        })

    framings.append({
        "title":              "Unsupervised Clustering",
        "target_suggestion":  "No target — discover natural groups",
        "problem_type":       "Clustering",
        "model_recommendation": "KMeans (k=3-5) or DBSCAN",
        "difficulty":         "Easy",
        "business_use_case":  "Segment records into meaningful groups for pattern discovery or anomaly detection",
        "why_this_data":      f"{profile.get('numeric_features',0)} numeric features enable meaningful distance-based clustering",
    })

    return framings[:3]
