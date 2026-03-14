import streamlit as st
import pandas as pd
from pathlib import Path
from src.debugger_engine import run_debugger_pipeline

# ----------------------------
# Page config
# ----------------------------
st.set_page_config(
    page_title="AutoML Debugger",
    page_icon="🧠",
    layout="centered"
)

# ----------------------------
# Header
# ----------------------------
st.title("🧠 AutoML Debugger")

st.markdown("""
## 🚀 Automated ML Dataset Diagnostics Platform

A lightweight tool that simulates how **ML engineers evaluate datasets before modeling or deployment**.

### 🔍 What this app does
- 📊 Computes quantitative ML health metrics  
- 🧼 Automatically cleans missing & mixed-type data  
- 🔡 Encodes categorical variables safely  
- 📈 Measures predictive signal strength  
- 🧠 Produces expert-level analysis in clear bullet points  

### 📦 What you get as output
- Model readiness metrics  
- Dataset quality diagnosis  
- Clear explanation of risks & strengths  
- Guidance on whether the data is ML-ready  
""")

st.divider()

# ----------------------------
# Dataset handling
# ----------------------------
FALLBACK_DATASET = Path("data/initial_dataset.csv")

def load_dataset(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "uploaded"
    if FALLBACK_DATASET.exists():
        return pd.read_csv(FALLBACK_DATASET), "fallback"
    return None, None

st.subheader("📂 Upload CSV Dataset (Optional)")
uploaded_file = st.file_uploader(
    "Upload a CSV file or run diagnostics using the default training dataset",
    type=["csv"]
)

df, source = load_dataset(uploaded_file)

if source == "uploaded":
    st.success("Using uploaded dataset")

elif source == "fallback":
    st.info("No dataset uploaded — using built-in training dataset")

# ----------------------------
# Target column
# ----------------------------
target_column = None

if df is not None:
    target_column = st.selectbox("🎯 Select Target Column", df.columns)

# ----------------------------
# Run debugger
# ----------------------------
if st.button("🚀 Run AutoML Debugger"):

    if df is None:

        st.info("No dataset available. Please upload a dataset.")

    else:

        with st.spinner("Running ML diagnostics..."):

            output = run_debugger_pipeline(df, target_column)

        # ----------------------------
        # Metrics
        # ----------------------------
        st.subheader("📊 Model Metrics")

        if output["metrics"]:

            st.json(output["metrics"])

        else:

            st.info("Metrics could not be computed for this dataset.")

        # ----------------------------
        # Expert analysis
        # ----------------------------
        st.subheader("🧠 Expert Analysis")

        for point in output["llm_analysis"]:

            st.markdown(f"- {point}")

        # ----------------------------
        # Dataset health score
        # ----------------------------
        st.subheader("⭐ Dataset Health Score")

        score = output["metrics"].get("dataset_health_score", 0)

        st.progress(score / 100)

        st.write(f"Dataset Health Score: {score}/100")

        # ----------------------------
        # Feature importance
        # ----------------------------
        st.subheader("📊 Feature Importance")

        importance = output.get("feature_importance", {})

        if importance:

            fi_df = pd.DataFrame({
                "Feature": list(importance.keys()),
                "Importance": list(importance.values())
            })

            st.bar_chart(fi_df.set_index("Feature"))

        else:

            st.info("Feature importance could not be computed for this dataset.")