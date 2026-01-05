# app.py
# Streamlit UI for AutoML Debugger

import streamlit as st
from src.debugger_engine import run_debugger_pipeline

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="AutoML Debugger",
    page_icon="🧠",
    layout="centered"
)

# -----------------------------
# APP TITLE & DESCRIPTION
# -----------------------------
st.title("🧠 AutoML Debugger")
st.write(
    """
    This application analyzes a machine learning pipeline and explains:
    - How the model is performing
    - Whether it is overfitting or underfitting
    - What potential issues exist
    - How to fix them (in simple language)
    """
)

st.divider()

# -----------------------------
# RUN BUTTON
# -----------------------------
if st.button("🚀 Run Debugger"):
    
    # Show loading spinner while analysis runs
    with st.spinner("Analyzing model and diagnosing issues..."):
        output = run_debugger_pipeline()

    st.success("Analysis completed successfully!")

    st.divider()

    # -----------------------------
    # MODEL PERFORMANCE
    # -----------------------------
    st.subheader("📊 Model Performance")
    st.write(f"**Training Accuracy:** {output['train_accuracy']}")
    st.write(f"**Validation Accuracy:** {output['validation_accuracy']}")

    st.divider()

    # -----------------------------
    # DIAGNOSIS SUMMARY
    # -----------------------------
    st.subheader("🩺 Diagnosis Summary")
    st.write(output["diagnosis"])

    st.divider()

    # -----------------------------
    # ROOT CAUSES & RECOMMENDATIONS
    # -----------------------------
    st.subheader("🔍 Root Causes & Recommendations")

    for i, cause in enumerate(output["root_causes"]):
        st.markdown(f"**Issue {i + 1}:** {cause}")
        st.markdown(f"👉 **Recommendation:** {output['recommendations'][i]}")
        st.write("")

    st.divider()

    # -----------------------------
    # HUMAN-FRIENDLY EXPLANATION
    # -----------------------------
    st.subheader("🧠 Human-Friendly Explanation")

    for explanation in output["explanations"]:
        st.info(explanation)
