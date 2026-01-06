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
# Run debugger (ALWAYS enabled)
# ----------------------------
if st.button("🚀 Run AutoML Debugger"):
    if df is None:
        st.info("No dataset available. Please upload a dataset or add a fallback dataset.")
    else:
        with st.spinner("Running ML diagnostics..."):
            output = run_debugger_pipeline(df, target_column)

        st.subheader("📊 Model Metrics")
        if output["metrics"]:
            st.json(output["metrics"])
        else:
            st.info("Metrics could not be computed for this dataset.")

        st.subheader("🧠 Expert Analysis")
        for point in output["llm_analysis"]:
            st.markdown(f"- {point}")
