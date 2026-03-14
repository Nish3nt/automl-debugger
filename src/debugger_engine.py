import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


# -----------------------------
# LLM-style analysis
# -----------------------------
def generate_llm_analysis(metrics, diagnosis):

    rows = metrics["rows"]
    cols = metrics["columns"]
    missing = metrics["missing_values"]
    num_feat = metrics["numeric_features"]
    cat_feat = metrics["categorical_features"]
    r2 = metrics["r2_score"]
    mae = metrics["mae"]

    insights = []

    insights.append(
        f"The dataset contains {rows} rows and {cols} columns."
    )

    if missing > 0:
        insights.append(
            f"{missing} missing values were detected and automatically handled."
        )
    else:
        insights.append(
            "The dataset does not contain missing values."
        )

    insights.append(
        f"{num_feat} numeric features were standardized for modeling stability."
    )

    if cat_feat > 0:
        insights.append(
            f"{cat_feat} categorical features were encoded using one-hot encoding."
        )

    if r2 > 0.8:
        insights.append(
            "The regression model shows strong predictive signal."
        )
    elif r2 > 0.5:
        insights.append(
            "The dataset shows moderate predictive capability."
        )
    else:
        insights.append(
            "The predictive signal appears relatively weak."
        )

    insights.append(
        f"The Mean Absolute Error of the model is {mae}."
    )

    insights.append(diagnosis)

    insights.append(
        "Overall the dataset appears suitable for machine learning experimentation."
    )

    return insights


# -----------------------------
# Main pipeline
# -----------------------------
def run_debugger_pipeline(df: pd.DataFrame, target_column: str | None):

    df = df.copy()

    # dataset too small
    if df.shape[0] < 5 or df.shape[1] < 2:
        return {
            "metrics": {},
            "diagnosis": "Dataset too small.",
            "llm_analysis": ["Dataset too small for ML diagnostics."],
            "feature_importance": {}
        }

    # target column selection
    if target_column is None or target_column not in df.columns:
        target_column = df.columns[-1]

    # -----------------------------
    # CLEAN TARGET COLUMN
    # -----------------------------

    # convert target to numeric
    df[target_column] = pd.to_numeric(df[target_column], errors="coerce")

    # drop rows where target is missing
    df = df.dropna(subset=[target_column])

    if df.shape[0] < 10:
        return {
            "metrics": {},
            "diagnosis": "Not enough valid target values.",
            "llm_analysis": [
                "Too many missing or invalid values in the target column."
            ],
            "feature_importance": {}
        }

    X = df.drop(columns=[target_column])
    y = df[target_column]

    # -----------------------------
    # feature detection
    # -----------------------------
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # -----------------------------
    # pipelines
    # -----------------------------
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])

    # -----------------------------
    # split
    # -----------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.25,
        random_state=42
    )

    # -----------------------------
    # model
    # -----------------------------
    model = Pipeline([
        ("preprocessing", preprocessor),
        ("regressor", LinearRegression())
    ])

    try:

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

    except Exception:

        return {
            "metrics": {},
            "diagnosis": "Model training failed.",
            "llm_analysis": [
                "The dataset structure prevented model training."
            ],
            "feature_importance": {}
        }

    # -----------------------------
    # feature importance
    # -----------------------------
    feature_importance = {}

    try:

        reg = model.named_steps["regressor"]

        coef = np.abs(reg.coef_[:len(numeric_features)])

        feature_importance = dict(
            sorted(
                zip(numeric_features, coef),
                key=lambda x: x[1],
                reverse=True
            )[:5]
        )

    except Exception:
        feature_importance = {}

    # -----------------------------
    # dataset health score
    # -----------------------------
    missing_ratio = df.isna().sum().sum() / (df.shape[0] * df.shape[1])

    score = 100 - int(missing_ratio * 40)

    if r2 < 0:
        score -= 40
    elif r2 < 0.3:
        score -= 25
    elif r2 < 0.6:
        score -= 10

    score = max(score, 0)

    metrics = {
        "r2_score": round(float(r2), 4),
        "mae": round(float(mae), 4),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_features": len(numeric_features),
        "categorical_features": len(categorical_features),
        "missing_values": int(df.isna().sum().sum()),
        "dataset_health_score": score
    }

    # diagnosis
    if r2 < 0:
        diagnosis = "Very weak predictive signal detected."
    elif r2 < 0.3:
        diagnosis = "Weak predictive signal detected."
    elif r2 < 0.6:
        diagnosis = "Moderate predictive signal detected."
    else:
        diagnosis = "Strong predictive signal detected."

    llm_analysis = generate_llm_analysis(metrics, diagnosis)

    return {
        "metrics": metrics,
        "diagnosis": diagnosis,
        "llm_analysis": llm_analysis,
        "feature_importance": feature_importance
    }