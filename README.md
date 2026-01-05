# 🧠 AutoML Debugger

AutoML Debugger is an end-to-end Machine Learning diagnostics system that focuses on explaining why a model behaves the way it does, rather than only reporting accuracy metrics.

## Overview
This project analyzes a machine learning pipeline and identifies issues related to data quality, training behavior, feature dominance, and model generalization. It performs root-cause analysis using deterministic ML logic and generates human-friendly explanations inspired by GenAI principles.

## Key Features
- Data quality analysis
- Overfitting and underfitting detection
- Feature dominance and correlation checks
- Rule-based root cause diagnosis
- Human-friendly explanations (GenAI-style, non-hallucinatory)
- Interactive Streamlit web application

## Architecture
Dataset → ML Model Training → Diagnostics Engine → Root Cause Analysis → Human-Friendly Explanation → Streamlit UI  
All diagnostic decisions are deterministic and explainable. GenAI is used only for explanation, not for decision-making.

## How to Run Locally
1. Create a virtual environment  
python -m venv venv  
venv\Scripts\activate  

2. Install dependencies  
pip install -r requirements.txt  

3. Run the application  
streamlit run app.py  

Open the app in browser at: http://localhost:8501

## Backend Test
python src/test_engine.py

## Tech Stack
Python, scikit-learn, Pandas, NumPy, Streamlit

## Why This Project Stands Out
Most ML projects stop at model accuracy. AutoML Debugger goes a step further by diagnosing model behavior, identifying root causes of issues, and explaining them in a human-readable way similar to real-world ML systems.

## Author
Nishant Diwate
