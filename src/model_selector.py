"""
AutoML Debugger — Model Selector  (Stage 2 v3.1 — Fast Edition)
================================================================
Key changes for speed:
  - No RandomizedSearchCV — uses battle-tested smart defaults instead
  - 3-fold CV (was 5) — 40% faster, still reliable
  - Tier 1 capped at 1,500 rows
  - n_jobs=1 everywhere — Streamlit Cloud is single-CPU
  - NaN/inf target rows dropped before any training
  - Target training time: < 60 seconds on any dataset

Models:
  Regression     : Ridge, XGBoost, LightGBM
  Classification : Logistic, XGBoost, LightGBM
  Clustering     : KMeans (elbow k=2-8)
  Time-series    : Ridge + XGBoost with lag features + chronological split
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Any, Callable

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import (
    train_test_split, cross_val_score, TimeSeriesSplit,
)
from sklearn.metrics import (
    r2_score, mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, roc_auc_score,
)
from sklearn.cluster import KMeans

try:
    from xgboost import XGBRegressor, XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMRegressor, LGBMClassifier
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────
# SMART DEFAULTS  (replace RandomizedSearchCV entirely)
# These are battle-tested defaults that work well on most tabular data.
# ─────────────────────────────────────────────────────────────────

SMART_DEFAULTS = {
    "regression": {
        "Ridge Regression": {"alpha": 1.0},
        "XGBoost": {
            "n_estimators": 200, "max_depth": 5,
            "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 3,
            "reg_alpha": 0.1, "reg_lambda": 1.0,
        },
        "LightGBM": {
            "n_estimators": 200, "max_depth": 5,
            "learning_rate": 0.05, "num_leaves": 31,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_samples": 10,
        },
    },
    "classification": {
        "Logistic Regression": {"C": 1.0, "max_iter": 500, "solver": "lbfgs"},
        "XGBoost": {
            "n_estimators": 200, "max_depth": 4,
            "learning_rate": 0.05, "subsample": 0.8,
            "colsample_bytree": 0.8, "min_child_weight": 3,
            "use_label_encoder": False, "eval_metric": "logloss",
        },
        "LightGBM": {
            "n_estimators": 200, "max_depth": 4,
            "learning_rate": 0.05, "num_leaves": 31,
            "subsample": 0.8, "colsample_bytree": 0.8,
            "min_child_samples": 10,
        },
    },
}


# ─────────────────────────────────────────────────────────────────
# PREPROCESSOR
# ─────────────────────────────────────────────────────────────────

def build_preprocessor(numeric_features: list, categorical_features: list) -> ColumnTransformer:
    transformers = []
    if numeric_features:
        transformers.append(("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ]), numeric_features))
    if categorical_features:
        transformers.append(("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_features))
    return ColumnTransformer(transformers, remainder="drop")


# ─────────────────────────────────────────────────────────────────
# LAG FEATURES  (time-series)
# ─────────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, target_column: str, lags: int = 3) -> pd.DataFrame:
    df = df.copy()
    for i in range(1, lags + 1):
        df[f"{target_column}_lag_{i}"] = df[target_column].shift(i)
    return df.dropna().reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE
# ─────────────────────────────────────────────────────────────────

def extract_feature_importance(pipeline, numeric_features, categorical_features) -> dict[str, float]:
    try:
        estimator = pipeline.named_steps["estimator"]
        prep      = pipeline.named_steps["preprocessing"]
        all_names: list[str] = []
        for name, _, cols in prep.transformers_:
            if name == "num":
                all_names.extend(cols)
            elif name == "cat":
                enc = prep.named_transformers_["cat"].named_steps["encoder"]
                all_names.extend(enc.get_feature_names_out(cols).tolist())
        if hasattr(estimator, "feature_importances_"):
            fi = estimator.feature_importances_
        elif hasattr(estimator, "coef_"):
            fi = np.abs(estimator.coef_).flatten()
        else:
            return {}
        pairs = sorted(zip(all_names, fi), key=lambda x: x[1], reverse=True)[:12]
        return {k: round(float(v), 5) for k, v in pairs}
    except Exception:
        return {}


# ─────────────────────────────────────────────────────────────────
# BASELINE
# ─────────────────────────────────────────────────────────────────

def compute_baseline(y_test, y_train, task_type: str) -> dict:
    if task_type == "regression":
        pred = np.full(len(y_test), y_train.mean())
        return {
            "model": "Baseline (mean predictor)",
            "r2":    round(float(r2_score(y_test, pred)), 4),
            "mae":   round(float(mean_absolute_error(y_test, pred)), 4),
            "description": f"Always predicts mean={float(y_train.mean()):.3f}",
        }
    else:
        majority = y_train.mode()[0]
        pred     = np.full(len(y_test), majority)
        return {
            "model":    "Baseline (majority class)",
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "description": f"Always predicts class '{majority}'",
        }


# ─────────────────────────────────────────────────────────────────
# CLEAN TARGET  (drop NaN / inf rows)
# ─────────────────────────────────────────────────────────────────

def clean_target(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Drop rows where target is NaN or infinite."""
    mask = y.notna() & ~np.isinf(y.replace([np.inf, -np.inf], np.nan).fillna(np.nan))
    n_dropped = (~mask).sum()
    return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True), int(n_dropped)


# ─────────────────────────────────────────────────────────────────
# MAKE ESTIMATOR
# ─────────────────────────────────────────────────────────────────

def make_estimator(name: str, task_type: str):
    defaults = SMART_DEFAULTS.get(task_type, {}).get(name, {})
    if task_type == "regression":
        if name == "Ridge Regression":
            return Ridge(**defaults)
        elif name == "XGBoost" and XGB_AVAILABLE:
            return XGBRegressor(random_state=42, verbosity=0, n_jobs=1, **defaults)
        elif name == "LightGBM" and LGB_AVAILABLE:
            return LGBMRegressor(random_state=42, verbosity=-1, n_jobs=1, **defaults)
    else:
        if name == "Logistic Regression":
            return LogisticRegression(random_state=42, n_jobs=1, **defaults)
        elif name == "XGBoost" and XGB_AVAILABLE:
            return XGBClassifier(random_state=42, verbosity=0, n_jobs=1, **defaults)
        elif name == "LightGBM" and LGB_AVAILABLE:
            return LGBMClassifier(random_state=42, verbosity=-1, n_jobs=1, **defaults)
    return None


# ─────────────────────────────────────────────────────────────────
# TIER 1 — FAST SCREENING  (sample, 3-fold, default params)
# ─────────────────────────────────────────────────────────────────

def tier1_screen(
    X: pd.DataFrame, y: pd.Series,
    task_type: str,
    numeric_features: list, categorical_features: list,
    is_timeseries: bool = False,
    progress_callback: Callable | None = None,
) -> list[dict]:
    """
    Screen all models on up to 1,500 rows with 3-fold CV.
    Target time: < 10 seconds total.
    """
    SAMPLE = 1500
    if len(X) > SAMPLE:
        idx = np.random.RandomState(42).choice(len(X), SAMPLE, replace=False)
        Xs  = X.iloc[idx].reset_index(drop=True)
        ys  = y.iloc[idx].reset_index(drop=True)
    else:
        Xs, ys = X.copy(), y.copy()

    le = None
    if task_type == "classification":
        le = LabelEncoder()
        ys = pd.Series(le.fit_transform(ys), name=ys.name)

    cv      = TimeSeriesSplit(n_splits=3) if is_timeseries else 3
    scoring = "r2" if task_type == "regression" else "accuracy"

    model_names = (
        ["Ridge Regression"] +
        (["XGBoost"] if XGB_AVAILABLE else []) +
        (["LightGBM"] if LGB_AVAILABLE else [])
        if task_type == "regression"
        else
        ["Logistic Regression"] +
        (["XGBoost"] if XGB_AVAILABLE else []) +
        (["LightGBM"] if LGB_AVAILABLE else [])
    )

    candidates = []
    for name in model_names:
        if progress_callback:
            progress_callback(f"⚡ Screening {name} (sample {SAMPLE} rows, 3-fold)...")
        try:
            est  = make_estimator(name, task_type)
            pipe = Pipeline([
                ("preprocessing", build_preprocessor(numeric_features, categorical_features)),
                ("estimator", est),
            ])
            scores = cross_val_score(pipe, Xs, ys, cv=cv, scoring=scoring, n_jobs=1)
            candidates.append({
                "model":    name,
                "cv_score": round(float(scores.mean()), 4),
                "cv_std":   round(float(scores.std()),  4),
            })
            if progress_callback:
                progress_callback(f"  → {name}: CV={scores.mean():.4f} (±{scores.std():.4f})")
        except Exception as e:
            candidates.append({"model": name, "cv_score": -999, "cv_std": 0, "error": str(e)})
            if progress_callback:
                progress_callback(f"  → {name}: failed ({str(e)[:60]})")

    candidates.sort(key=lambda x: x["cv_score"], reverse=True)
    return candidates


# ─────────────────────────────────────────────────────────────────
# TIER 2 — FULL TRAINING  (smart defaults, 3-fold CV, full data)
# ─────────────────────────────────────────────────────────────────

def tier2_train(
    X: pd.DataFrame, y: pd.Series,
    task_type: str,
    numeric_features: list, categorical_features: list,
    tier1_results: list[dict],
    is_timeseries: bool = False,
    progress_callback: Callable | None = None,
) -> dict[str, Any]:
    """
    Full training of top 2 Tier-1 models using smart defaults.
    No RandomizedSearchCV — smart defaults are used directly.
    Target time: < 45 seconds total.
    """
    le = None
    if task_type == "classification":
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(y), name=y.name)

    # Train/test split
    if is_timeseries:
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]
        cv = TimeSeriesSplit(n_splits=3)
    else:
        strat = y if task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat,
        )
        cv = 3

    scoring = "r2" if task_type == "regression" else "accuracy"
    baseline = compute_baseline(y_test, y_train, task_type)

    top2 = [r for r in tier1_results if r.get("cv_score", -999) > -999][:2]
    leaderboard: list[dict] = []
    best_pipeline  = None
    best_score     = -np.inf
    feature_importance: dict[str, float] = {}

    for candidate in top2:
        name = candidate["model"]
        if progress_callback:
            progress_callback(f"🔬 Full training: {name} on {len(X):,} rows (smart defaults)...")

        try:
            est  = make_estimator(name, task_type)
            if est is None:
                continue

            pipe = Pipeline([
                ("preprocessing", build_preprocessor(numeric_features, categorical_features)),
                ("estimator", est),
            ])

            # Fit on train, CV on full dataset
            pipe.fit(X_train, y_train)
            cv_scores   = cross_val_score(pipe, X, y, cv=cv, scoring=scoring, n_jobs=1)
            cv_score    = round(float(cv_scores.mean()), 4)

            y_pred_train = pipe.predict(X_train)
            y_pred_test  = pipe.predict(X_test)

            if task_type == "regression":
                train_score = round(float(r2_score(y_train, y_pred_train)), 4)
                test_r2     = round(float(r2_score(y_test,  y_pred_test)),  4)
                mae         = round(float(mean_absolute_error(y_test, y_pred_test)), 4)
                rmse        = round(float(np.sqrt(mean_squared_error(y_test, y_pred_test))), 4)
                pred_std    = round(float((y_test.values - y_pred_test).std()), 4)

                result = {
                    "model": name, "cv_score": cv_score,
                    "train_score": train_score, "test_r2": test_r2,
                    "mae": mae, "rmse": rmse, "pred_std": pred_std,
                    "pipeline": pipe, "le": le,
                }
            else:
                avg = "binary" if y.nunique() == 2 else "weighted"
                train_score = round(float(accuracy_score(y_train, y_pred_train)), 4)
                test_acc    = round(float(accuracy_score(y_test,  y_pred_test)),  4)
                f1          = round(float(f1_score(y_test, y_pred_test, average=avg, zero_division=0)), 4)
                auc         = None
                if y.nunique() == 2:
                    try:
                        proba = pipe.predict_proba(X_test)[:, 1]
                        auc   = round(float(roc_auc_score(y_test, proba)), 4)
                    except Exception:
                        pass

                result = {
                    "model": name, "cv_score": cv_score,
                    "train_score": train_score, "test_accuracy": test_acc,
                    "f1_score": f1, "roc_auc": auc,
                    "pipeline": pipe, "le": le,
                }

            leaderboard.append(result)

            if cv_score > best_score:
                best_score = cv_score
                best_pipeline = pipe
                feature_importance = extract_feature_importance(pipe, numeric_features, categorical_features)

            if progress_callback:
                progress_callback(f"  ✅ {name}: CV={cv_score:.4f} | Train={train_score:.4f}")

        except Exception as e:
            leaderboard.append({"model": name, "cv_score": -999, "error": str(e)})
            if progress_callback:
                progress_callback(f"  ❌ {name} failed: {str(e)[:80]}")

    leaderboard.sort(key=lambda x: x.get("cv_score", -999), reverse=True)

    return {
        "leaderboard":         leaderboard,
        "best_pipeline":       best_pipeline,
        "feature_importance":  feature_importance,
        "baseline":            baseline,
        "label_encoder":       le,
        "X_test":              X_test,
        "y_test":              y_test,
        "task_type":           task_type,
    }


# ─────────────────────────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────────────────────────

def run_clustering(X: pd.DataFrame, numeric_features: list,
                   progress_callback: Callable | None = None) -> dict[str, Any]:
    if progress_callback:
        progress_callback("🔵 Running KMeans (k=2 to 8)...")

    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    X_num    = X[numeric_features] if numeric_features else X.select_dtypes(include=[np.number])
    X_scaled = prep.fit_transform(X_num)

    inertias: list[float] = []
    k_range  = range(2, min(9, len(X) // 10 + 2))

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))

    if len(inertias) >= 3:
        drops  = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        diffs2 = [drops[i] - drops[i+1] for i in range(len(drops)-1)]
        best_k = list(k_range)[diffs2.index(max(diffs2)) + 1]
    else:
        best_k = 2

    if progress_callback:
        progress_callback(f"✅ Optimal k={best_k}")

    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels  = km_best.fit_predict(X_scaled)

    X_orig           = X_num.copy()
    X_orig["Cluster"] = labels
    profiles         = X_orig.groupby("Cluster").mean().round(3).reset_index()
    cluster_sizes    = pd.Series(labels).value_counts().sort_index().to_dict()

    return {
        "best_k":        best_k,
        "labels":        labels.tolist(),
        "cluster_sizes": {f"Cluster {k}": v for k, v in cluster_sizes.items()},
        "profiles":      profiles,
        "elbow_data":    {"k_values": list(k_range), "inertias": inertias},
        "inertia":       float(km_best.inertia_),
    }


# ─────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────

def run_model_pipeline(
    df:                pd.DataFrame,
    target_column:     str | None,
    task_type:         str,
    is_timeseries:     bool = False,
    ts_datetime_col:   str | None = None,
    progress_callback: Callable | None = None,
    api_key:           str | None = None,
) -> dict[str, Any]:

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # ── CLUSTERING ───────────────────────────
    if target_column is None or task_type == "clustering":
        log("🔵 Clustering mode...")
        X = df.copy()
        if ts_datetime_col and ts_datetime_col in X.columns:
            X = X.drop(columns=[ts_datetime_col])
        X   = X.select_dtypes(include=[np.number])
        nf  = X.columns.tolist()
        res = run_clustering(X, nf, progress_callback=log)
        res["mode"] = "clustering"
        return res

    # ── SUPERVISED ───────────────────────────
    log(f"📋 Task: {task_type.upper()}")

    df_model = df.copy()

    # Drop datetime column
    if ts_datetime_col and ts_datetime_col in df_model.columns and ts_datetime_col != target_column:
        df_model = df_model.drop(columns=[ts_datetime_col])
        log(f"🗓️ Dropped datetime column '{ts_datetime_col}'")

    # Lag features for time-series
    if is_timeseries and task_type == "regression":
        log("⏱️ Adding lag features (3 lags)...")
        df_model = add_lag_features(df_model, target_column, lags=3)
        log(f"✅ Shape after lags: {df_model.shape}")

    # Drop constant columns
    const = [c for c in df_model.columns if c != target_column and df_model[c].nunique() <= 1]
    if const:
        df_model.drop(columns=const, inplace=True)

    # Coerce target
    if task_type == "regression":
        df_model[target_column] = pd.to_numeric(df_model[target_column], errors="coerce")

    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]

    # ── DROP NaN/INF TARGET ROWS ─────────────
    X, y, n_dropped = clean_target(X, y)
    if n_dropped > 0:
        log(f"🧹 Dropped {n_dropped} rows with invalid target values.")
    if len(X) < 20:
        return {"mode": "supervised", "error": "Not enough valid rows after cleaning target."}

    # Drop date-like string columns
    date_like = []
    for col in X.select_dtypes(exclude=[np.number]).columns:
        try:
            parsed = pd.to_datetime(X[col], format="mixed", dayfirst=False)
            if parsed.notna().mean() > 0.9:
                date_like.append(col)
        except Exception:
            pass
    if date_like:
        X = X.drop(columns=date_like)
        log(f"🗓️ Dropped {len(date_like)} date-like column(s): {date_like}")

    nf = X.select_dtypes(include=[np.number]).columns.tolist()
    cf = X.select_dtypes(exclude=[np.number]).columns.tolist()

    log(f"📊 {len(nf)} numeric + {len(cf)} categorical features | {len(X):,} rows")

    # TIER 1
    log("⚡ Tier 1 — fast screening...")
    tier1 = tier1_screen(X, y, task_type, nf, cf,
                          is_timeseries=is_timeseries,
                          progress_callback=log)

    best_t1 = tier1[0] if tier1 else {}
    log(f"✅ Tier 1 leader: {best_t1.get('model','?')} (CV={best_t1.get('cv_score','?')})")

    # TIER 2
    log("🔬 Tier 2 — full training with smart defaults...")
    result = tier2_train(X, y, task_type, nf, cf,
                          tier1_results=tier1,
                          is_timeseries=is_timeseries,
                          progress_callback=log)

    result["mode"]                 = "supervised"
    result["numeric_features"]     = nf
    result["categorical_features"] = cf
    result["tier1_results"]        = tier1

    best = result["leaderboard"][0] if result["leaderboard"] else {}
    log(f"🏆 Winner: {best.get('model','?')} — CV={best.get('cv_score','?')}")
    log("✅ Pipeline complete!")

    return result
