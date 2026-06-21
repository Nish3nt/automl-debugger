"""
AutoML Debugger — Model Selector  (Stage 1 v3.0)
==================================================
Two-tier training:
  Tier 1 — Fast screening on sample (3-5 sec)
  Tier 2 — Full train + RandomizedSearchCV on top 2 models

Models:
  Regression     : Ridge, XGBoost, LightGBM
  Classification : Logistic, XGBoost, LightGBM
  Clustering     : KMeans (elbow method, k=2–8)
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
    train_test_split, cross_val_score,
    TimeSeriesSplit, RandomizedSearchCV,
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
# PREPROCESSOR BUILDER
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
# LAG FEATURE ENGINEERING  (time-series only)
# ─────────────────────────────────────────────────────────────────

def add_lag_features(df: pd.DataFrame, target_column: str, lags: int = 5) -> pd.DataFrame:
    """
    Creates lag features for the target column so tree models can
    learn temporal patterns without a native time-series model.
    """
    df = df.copy()
    for i in range(1, lags + 1):
        df[f"{target_column}_lag_{i}"] = df[target_column].shift(i)
    df = df.dropna().reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────
# FEATURE IMPORTANCE EXTRACTOR
# ─────────────────────────────────────────────────────────────────

def extract_feature_importance(
    model_pipeline: Pipeline,
    numeric_features: list,
    categorical_features: list,
) -> dict[str, float]:
    try:
        estimator = model_pipeline.named_steps["estimator"]
        prep      = model_pipeline.named_steps["preprocessing"]

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
# BASELINE COMPARISON
# ─────────────────────────────────────────────────────────────────

def compute_baseline(y_test: pd.Series, y_train: pd.Series, task_type: str) -> dict:
    """
    Dumbest possible baseline:
      Regression     → always predict training mean
      Classification → always predict majority class
    """
    if task_type == "regression":
        pred    = np.full(len(y_test), y_train.mean())
        r2      = round(float(r2_score(y_test, pred)), 4)
        mae     = round(float(mean_absolute_error(y_test, pred)), 4)
        return {"model": "Baseline (mean predictor)", "r2": r2, "mae": mae,
                "description": f"Always predicts mean={y_train.mean():.3f}"}
    else:
        majority = y_train.mode()[0]
        pred     = np.full(len(y_test), majority)
        acc      = round(float(accuracy_score(y_test, pred)), 4)
        return {"model": "Baseline (majority class)", "accuracy": acc,
                "description": f"Always predicts class '{majority}'"}


# ─────────────────────────────────────────────────────────────────
# TIER 1 — FAST SCREENING
# ─────────────────────────────────────────────────────────────────

def tier1_screen(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    numeric_features: list,
    categorical_features: list,
    is_timeseries: bool = False,
    progress_callback: Callable | None = None,
) -> list[dict]:
    """
    Runs all models with default settings on up to 5,000 rows.
    Returns preliminary leaderboard sorted by CV score.
    """
    sample_size = min(5000, len(X))
    if len(X) > sample_size:
        idx = np.random.choice(len(X), sample_size, replace=False)
        Xs, ys = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True)
    else:
        Xs, ys = X, y

    le = None
    if task_type == "classification":
        le = LabelEncoder()
        ys = pd.Series(le.fit_transform(ys), name=ys.name)

    cv = TimeSeriesSplit(n_splits=5) if is_timeseries else 5
    scoring = "r2" if task_type == "regression" else "accuracy"

    candidates = []

    if task_type == "regression":
        model_defs = [
            ("Ridge Regression", Ridge(alpha=1.0)),
        ]
        if XGB_AVAILABLE:
            model_defs.append(("XGBoost", XGBRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, verbosity=0, n_jobs=-1,
            )))
        if LGB_AVAILABLE:
            model_defs.append(("LightGBM", LGBMRegressor(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, verbosity=-1, n_jobs=-1,
            )))
    else:
        model_defs = [
            ("Logistic Regression", LogisticRegression(max_iter=500, random_state=42, n_jobs=-1)),
        ]
        if XGB_AVAILABLE:
            model_defs.append(("XGBoost", XGBClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, verbosity=0, use_label_encoder=False,
                eval_metric="logloss", n_jobs=-1,
            )))
        if LGB_AVAILABLE:
            model_defs.append(("LightGBM", LGBMClassifier(
                n_estimators=100, max_depth=5, learning_rate=0.1,
                random_state=42, verbosity=-1, n_jobs=-1,
            )))

    for name, estimator in model_defs:
        if progress_callback:
            progress_callback(f"⚡ Tier 1 — screening {name}...")
        try:
            pipe = Pipeline([
                ("preprocessing", build_preprocessor(numeric_features, categorical_features)),
                ("estimator", estimator),
            ])
            scores = cross_val_score(pipe, Xs, ys, cv=cv, scoring=scoring, n_jobs=-1)
            candidates.append({
                "model":      name,
                "cv_score":   round(float(scores.mean()), 4),
                "cv_std":     round(float(scores.std()),  4),
                "train_score": None,
                "tier":       1,
                "estimator":  estimator,
            })
        except Exception as e:
            candidates.append({"model": name, "cv_score": -999, "cv_std": 0,
                               "train_score": None, "tier": 1, "error": str(e)})

    candidates.sort(key=lambda x: x["cv_score"], reverse=True)
    return candidates


# ─────────────────────────────────────────────────────────────────
# TIER 2 — FULL TRAINING + HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────

XGB_REG_GRID = {
    "estimator__n_estimators":  [100, 200, 300],
    "estimator__max_depth":     [3, 5, 7],
    "estimator__learning_rate": [0.05, 0.1, 0.2],
    "estimator__subsample":     [0.8, 1.0],
    "estimator__colsample_bytree": [0.8, 1.0],
}
XGB_CLF_GRID = XGB_REG_GRID.copy()

LGB_REG_GRID = {
    "estimator__n_estimators":  [100, 200, 300],
    "estimator__max_depth":     [3, 5, 7],
    "estimator__learning_rate": [0.05, 0.1, 0.2],
    "estimator__num_leaves":    [31, 63, 127],
}
LGB_CLF_GRID = LGB_REG_GRID.copy()

RIDGE_GRID = {"estimator__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}
LOGISTIC_GRID = {"estimator__C": [0.01, 0.1, 1.0, 10.0], "estimator__solver": ["lbfgs", "liblinear"]}

PARAM_GRIDS = {
    "Ridge Regression":    RIDGE_GRID,
    "Logistic Regression": LOGISTIC_GRID,
    "XGBoost":             XGB_REG_GRID,
    "LightGBM":            LGB_REG_GRID,
}


def tier2_train(
    X: pd.DataFrame,
    y: pd.Series,
    task_type: str,
    numeric_features: list,
    categorical_features: list,
    tier1_results: list[dict],
    is_timeseries: bool = False,
    progress_callback: Callable | None = None,
) -> dict[str, Any]:
    """
    Full training on complete dataset for top 2 models from Tier 1.
    Runs RandomizedSearchCV (10 iterations) for tuning.
    Returns full leaderboard + best pipeline + feature importance.
    """
    le = None
    if task_type == "classification":
        le = LabelEncoder()
        y  = pd.Series(le.fit_transform(y), name=y.name)

    # Train/test split
    if is_timeseries:
        split_idx     = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        cv = TimeSeriesSplit(n_splits=5)
    else:
        strat = y if task_type == "classification" else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=strat,
        )
        cv = 5

    scoring = "r2" if task_type == "regression" else "accuracy"

    # Baseline
    baseline = compute_baseline(y_test, y_train, task_type)

    top2 = [r for r in tier1_results if r.get("cv_score", -999) > -999][:2]
    leaderboard: list[dict] = []
    best_pipeline = None
    best_score    = -np.inf
    feature_importance: dict[str, float] = {}

    for rank, candidate in enumerate(top2):
        name = candidate["model"]
        if progress_callback:
            progress_callback(f"🔬 Tier 2 — tuning {name} on full dataset ({len(X):,} rows)...")

        try:
            estimator = candidate["estimator"]

            # Re-instantiate fresh estimator for full training
            if task_type == "regression":
                if name == "Ridge Regression":  estimator = Ridge()
                elif name == "XGBoost" and XGB_AVAILABLE:
                    estimator = XGBRegressor(random_state=42, verbosity=0, n_jobs=-1)
                elif name == "LightGBM" and LGB_AVAILABLE:
                    estimator = LGBMRegressor(random_state=42, verbosity=-1, n_jobs=-1)
            else:
                if name == "Logistic Regression":
                    estimator = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
                elif name == "XGBoost" and XGB_AVAILABLE:
                    estimator = XGBClassifier(random_state=42, verbosity=0,
                                              use_label_encoder=False, eval_metric="logloss", n_jobs=-1)
                elif name == "LightGBM" and LGB_AVAILABLE:
                    estimator = LGBMClassifier(random_state=42, verbosity=-1, n_jobs=-1)

            pipe = Pipeline([
                ("preprocessing", build_preprocessor(numeric_features, categorical_features)),
                ("estimator", estimator),
            ])

            # Hyperparameter search
            param_grid = PARAM_GRIDS.get(name, {})
            if param_grid:
                search = RandomizedSearchCV(
                    pipe, param_grid, n_iter=10, cv=cv,
                    scoring=scoring, random_state=42, n_jobs=-1,
                )
                search.fit(X_train, y_train)
                best_pipe = search.best_estimator_
                cv_score  = round(float(search.best_score_), 4)
            else:
                pipe.fit(X_train, y_train)
                best_pipe = pipe
                scores    = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring)
                cv_score  = round(float(scores.mean()), 4)

            # Train & test scores
            y_pred_train = best_pipe.predict(X_train)
            y_pred_test  = best_pipe.predict(X_test)

            if task_type == "regression":
                train_score = round(float(r2_score(y_train, y_pred_train)), 4)
                test_r2     = round(float(r2_score(y_test,  y_pred_test)),  4)
                mae         = round(float(mean_absolute_error(y_test, y_pred_test)), 4)
                rmse        = round(float(np.sqrt(mean_squared_error(y_test, y_pred_test))), 4)

                # Prediction interval via residual std
                residuals = y_test.values - y_pred_test
                pred_std  = round(float(residuals.std()), 4)

                result = {
                    "model":       name,
                    "cv_score":    cv_score,
                    "train_score": train_score,
                    "test_r2":     test_r2,
                    "mae":         mae,
                    "rmse":        rmse,
                    "pred_std":    pred_std,
                    "tier":        2,
                    "pipeline":    best_pipe,
                    "le":          le,
                }

            else:
                avg = "binary" if y.nunique() == 2 else "weighted"
                train_score = round(float(accuracy_score(y_train, y_pred_train)), 4)
                test_acc    = round(float(accuracy_score(y_test,  y_pred_test)),  4)
                f1          = round(float(f1_score(y_test, y_pred_test, average=avg, zero_division=0)), 4)
                auc         = None
                if y.nunique() == 2:
                    try:
                        proba = best_pipe.predict_proba(X_test)[:, 1]
                        auc   = round(float(roc_auc_score(y_test, proba)), 4)
                    except Exception:
                        pass

                result = {
                    "model":       name,
                    "cv_score":    cv_score,
                    "train_score": train_score,
                    "test_accuracy": test_acc,
                    "f1_score":    f1,
                    "roc_auc":     auc,
                    "tier":        2,
                    "pipeline":    best_pipe,
                    "le":          le,
                }

            leaderboard.append(result)

            if cv_score > best_score:
                best_score    = cv_score
                best_pipeline = best_pipe
                feature_importance = extract_feature_importance(
                    best_pipe, numeric_features, categorical_features
                )

            if progress_callback:
                progress_callback(f"✅ {name} — CV={cv_score:.4f}, Train={train_score:.4f}")

        except Exception as e:
            leaderboard.append({"model": name, "cv_score": -999, "error": str(e), "tier": 2})

    leaderboard.sort(key=lambda x: x.get("cv_score", -999), reverse=True)

    return {
        "leaderboard":        leaderboard,
        "best_pipeline":      best_pipeline,
        "feature_importance": feature_importance,
        "baseline":           baseline,
        "label_encoder":      le,
        "X_test":             X_test,
        "y_test":             y_test,
        "task_type":          task_type,
    }


# ─────────────────────────────────────────────────────────────────
# CLUSTERING  (KMeans, elbow method)
# ─────────────────────────────────────────────────────────────────

def run_clustering(
    X: pd.DataFrame,
    numeric_features: list,
    progress_callback: Callable | None = None,
) -> dict[str, Any]:
    """
    Runs KMeans for k=2 to 8, picks best k via elbow (inertia drop).
    Returns cluster assignments, profiles, and elbow data.
    """
    if progress_callback:
        progress_callback("🔵 Running KMeans clustering (k=2 to 8)...")

    # Preprocess numeric only for clustering
    prep = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    X_num = X[numeric_features] if numeric_features else X.select_dtypes(include=[np.number])
    X_scaled = prep.fit_transform(X_num)

    inertias: list[float] = []
    k_range = range(2, min(9, len(X) // 10 + 2))

    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X_scaled)
        inertias.append(float(km.inertia_))

    # Elbow: pick k where marginal inertia drop is smallest (2nd derivative)
    if len(inertias) >= 3:
        drops  = [inertias[i] - inertias[i+1] for i in range(len(inertias)-1)]
        diffs2 = [drops[i] - drops[i+1] for i in range(len(drops)-1)]
        best_k = list(k_range)[diffs2.index(max(diffs2)) + 1]
    else:
        best_k = 2

    if progress_callback:
        progress_callback(f"✅ Optimal clusters: k={best_k} (elbow method)")

    km_best = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels  = km_best.fit_predict(X_scaled)

    # Cluster profiles (mean of original numeric features per cluster)
    X_orig = X[numeric_features].copy() if numeric_features else X.select_dtypes(include=[np.number]).copy()
    X_orig["Cluster"] = labels
    profiles = X_orig.groupby("Cluster").mean().round(3).reset_index()

    cluster_sizes = pd.Series(labels).value_counts().sort_index().to_dict()

    return {
        "best_k":         best_k,
        "labels":         labels.tolist(),
        "cluster_sizes":  {f"Cluster {k}": v for k, v in cluster_sizes.items()},
        "profiles":       profiles,
        "elbow_data": {
            "k_values":  list(k_range),
            "inertias":  inertias,
        },
        "inertia": float(km_best.inertia_),
    }


# ─────────────────────────────────────────────────────────────────
# MAIN ENTRY — run_model_pipeline
# ─────────────────────────────────────────────────────────────────

def run_model_pipeline(
    df:                  pd.DataFrame,
    target_column:       str | None,
    task_type:           str,
    is_timeseries:       bool = False,
    ts_datetime_col:     str | None = None,
    progress_callback:   Callable | None = None,
    api_key:             str | None = None,
) -> dict[str, Any]:
    """
    Full model pipeline entry point.

    If target_column is None → run clustering.
    Otherwise → run regression or classification.
    """

    def log(msg):
        if progress_callback:
            progress_callback(msg)

    # ── CLUSTERING MODE ──────────────────────────────────────────
    if target_column is None or task_type == "clustering":
        log("🔵 No target column — switching to KMeans clustering mode...")
        X = df.copy()
        if ts_datetime_col and ts_datetime_col in X.columns:
            X = X.drop(columns=[ts_datetime_col])
        X = X.select_dtypes(include=[np.number])
        numeric_features = X.columns.tolist()
        result = run_clustering(X, numeric_features, progress_callback=log)
        result["mode"] = "clustering"
        return result

    # ── SUPERVISED MODE ──────────────────────────────────────────
    log(f"📋 Task detected: {task_type.upper()}")

    df_model = df.copy()

    # Drop datetime column from features
    if ts_datetime_col and ts_datetime_col in df_model.columns and ts_datetime_col != target_column:
        df_model = df_model.drop(columns=[ts_datetime_col])
        log(f"⏱️ Dropped datetime column '{ts_datetime_col}' from features.")

    # Add lag features for time-series
    if is_timeseries and task_type == "regression":
        log("⏱️ Adding lag features for time-series...")
        df_model = add_lag_features(df_model, target_column, lags=5)
        log(f"✅ Added 5 lag features. New shape: {df_model.shape}")

    # Drop constant columns
    const_cols = [c for c in df_model.columns if c != target_column and df_model[c].nunique() <= 1]
    if const_cols:
        df_model.drop(columns=const_cols, inplace=True)
        log(f"🗑️ Dropped {len(const_cols)} constant column(s).")

    # Coerce target
    if task_type == "regression":
        df_model[target_column] = pd.to_numeric(df_model[target_column], errors="coerce")
    df_model = df_model.dropna(subset=[target_column]).reset_index(drop=True)

    if len(df_model) < 20:
        return {"mode": "supervised", "error": "Not enough valid rows after preprocessing."}

    X = df_model.drop(columns=[target_column])
    y = df_model[target_column]

    # Drop date-like string columns — OHE on dates produces useless sparse features
    date_like_cols = []
    for col in X.select_dtypes(exclude=[np.number]).columns:
        try:
            parsed = pd.to_datetime(X[col], format="mixed", dayfirst=False)
            if parsed.notna().mean() > 0.9:
                date_like_cols.append(col)
        except Exception:
            pass
    if date_like_cols:
        X = X.drop(columns=date_like_cols)
        log(f"🗓️ Dropped {len(date_like_cols)} date-like column(s): {date_like_cols}")

    numeric_features     = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    log(f"📊 Features: {len(numeric_features)} numeric, {len(categorical_features)} categorical")
    log(f"📏 Dataset: {len(X):,} rows for training")

    # TIER 1 — fast screening
    log("⚡ Starting Tier 1 — fast model screening on sample...")
    tier1 = tier1_screen(
        X, y, task_type, numeric_features, categorical_features,
        is_timeseries=is_timeseries,
        progress_callback=log,
    )
    log(f"✅ Tier 1 complete. Preliminary leader: {tier1[0]['model']} (CV={tier1[0]['cv_score']:.4f})")

    # TIER 2 — full training
    log("🔬 Starting Tier 2 — full training with hyperparameter tuning...")
    result = tier2_train(
        X, y, task_type, numeric_features, categorical_features,
        tier1_results=tier1,
        is_timeseries=is_timeseries,
        progress_callback=log,
    )

    result["mode"]                = "supervised"
    result["numeric_features"]    = numeric_features
    result["categorical_features"] = categorical_features
    result["tier1_results"]       = tier1

    best = result["leaderboard"][0] if result["leaderboard"] else {}
    log(f"🏆 Best model: {best.get('model','?')} — CV={best.get('cv_score','?')}")
    log("✅ Pipeline complete!")

    return result
