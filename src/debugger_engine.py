import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


def run_debugger_pipeline(df: pd.DataFrame, target_column: str | None):
    """
    Fully robust AutoML debugger pipeline.
    Compatible with ALL sklearn versions.
    """

    df = df.copy()

    # ----------------------------
    # Safety: minimal dataset
    # ----------------------------
    if df.shape[0] < 5 or df.shape[1] < 2:
        return {
            "metrics": {},
            "diagnosis": "Dataset too small for ML diagnostics.",
            "llm_analysis": [
                "The dataset does not contain enough samples or features.",
                "Machine learning models require more data to extract patterns.",
                "Consider collecting additional data."
            ]
        }

    # ----------------------------
    # Target handling
    # ----------------------------
    if target_column is None or target_column not in df.columns:
        target_column = df.columns[-1]

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # ----------------------------
    # Feature detection
    # ----------------------------
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # ----------------------------
    # Pipelines
    # ----------------------------
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    # 🔥 CRITICAL FIX: NO sparse / sparse_output ARG
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features)
        ],
        remainder="drop"
    )

    # ----------------------------
    # Train-test split
    # ----------------------------
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
    except Exception:
        return {
            "metrics": {},
            "diagnosis": "Train-test split failed due to data inconsistencies.",
            "llm_analysis": [
                "The dataset structure prevented proper splitting.",
                "This often occurs with severe data corruption.",
                "Verify column consistency and missing values."
            ]
        }

    # ----------------------------
    # Model
    # ----------------------------
    model = Pipeline(
        steps=[
            ("preprocessing", preprocessor),
            ("regressor", LinearRegression())
        ]
    )

    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
    except Exception:
        return {
            "metrics": {},
            "diagnosis": "Model training failed due to incompatible data.",
            "llm_analysis": [
                "The target variable may not be suitable for regression.",
                "Strong noise or non-numeric targets may exist.",
                "Consider transforming or redefining the target."
            ]
        }

    # ----------------------------
    # Metrics
    # ----------------------------
    metrics = {
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 4),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_features": len(numeric_features),
        "categorical_features": len(categorical_features),
        "missing_values": int(df.isna().sum().sum())
    }

    # ----------------------------
    # Diagnosis
    # ----------------------------
    if r2 < 0:
        diagnosis = "Very weak predictive signal detected."
    elif r2 < 0.3:
        diagnosis = "Weak predictive signal detected."
    elif r2 < 0.6:
        diagnosis = "Moderate predictive signal detected."
    else:
        diagnosis = "Strong predictive signal detected."

    # ----------------------------
    # LLM-style expert analysis
    # ----------------------------
    llm_analysis = [
        f"The dataset contains {metrics['rows']} rows and {metrics['columns']} columns.",
        f"{metrics['missing_values']} missing values were automatically handled.",
        f"{metrics['numeric_features']} numeric features were scaled for stability.",
        f"{metrics['categorical_features']} categorical features were encoded safely.",
        f"R² score of {metrics['r2_score']} reflects predictive signal strength.",
        f"MAE of {metrics['mae']} shows average prediction error.",
        diagnosis,
        "This analysis helps assess dataset readiness for ML modeling."
    ]

    return {
        "metrics": metrics,
        "diagnosis": diagnosis,
        "llm_analysis": llm_analysis
    }
