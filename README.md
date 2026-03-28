# 🧠 AutoML Debugger
### LLM-Assisted Dataset Diagnostics for Machine Learning Engineers

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Anthropic_Claude-Powered-5A4FCF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center">
  <b>Know whether your dataset is worth training on — before you waste compute.</b>
</p>

<p align="center">
  <a href="https://jzpahqig75a375i8ycpiy2.streamlit.app/">🚀 Live Demo</a> ·
  <a href="#-quick-start">Quick Start</a> ·
  <a href="#-features">Features</a> ·
  <a href="#-architecture">Architecture</a>
</p>

---

## 📌 What Is This?

**AutoML Debugger** is an industry-grade ML diagnostics tool that automatically evaluates dataset quality, detects weak predictive signals, and explains results using real **LLM-powered expert reasoning** via Anthropic Claude.

Inspired by internal data validation tools used at ML teams before model deployment — this tool answers the questions that matter **before** you train anything:

| Question | Answer from AutoML Debugger |
|---|---|
| Does my dataset have signal? | ✅ R², F1, AUC-ROC, cross-validated scores |
| Is the data clean enough? | ✅ Missing values, duplicates, outliers |
| Are there hidden data traps? | ✅ Class imbalance, constant features, high cardinality |
| What should I fix first? | ✅ LLM bullet-point analysis with exact numbers |
| Is this production-ready? | ✅ Dataset health score 0–100 |

---

## ✨ Features

### 📊 Auto Task Detection
Automatically detects whether your problem is **regression** or **classification** from the target column — no manual configuration needed.

### 🤖 Real LLM Analysis (Not Hardcoded Text)
Passes actual computed metrics to **Anthropic Claude** which returns expert-level diagnosis. Falls back to rule-based analysis when no API key is provided.

### 🌲 Dual-Model Evaluation
Runs both a **baseline interpretable model** (Linear/Logistic Regression) and a **Random Forest** for a fair performance comparison.

### 📈 Rich Metric Suite
- **Regression:** R² (baseline + RF), MAE, RMSE, 5-fold CV R²
- **Classification:** Accuracy (baseline + RF), F1 Score, ROC-AUC, 5-fold CV Accuracy

### 🔍 Data Quality Profiler
- Missing value percentage per column
- Duplicate row detection
- IQR-based outlier detection per feature
- Class imbalance ratio
- Constant / near-constant feature flags
- High-cardinality categorical column detection

### 📉 Interactive Visualizations (Plotly)
- Radar chart for overall dataset quality dimensions
- Feature importance horizontal bar chart
- Pearson correlation with target
- Missing value bar chart
- Target distribution (pie for classification, histogram for regression)

### ⭐ Dataset Health Score
A single 0–100 composite score penalising missing data, duplicates, imbalance, constant features, and weak model signal.

---



> Without a key the app still runs fully — it uses rule-based analysis instead of Claude.



---

## 🏗️ Architecture

```
CSV Upload / Fallback Dataset
         │
         ▼
  ┌─────────────────┐
  │  Data Profiler  │  ← missing values, duplicates, outliers,
  └────────┬────────┘    class imbalance, constant features
           │
           ▼
  ┌─────────────────────────────┐
  │  Task Auto-Detection        │  ← regression vs. classification
  └────────────┬────────────────┘
               │
       ┌───────┴───────┐
       ▼               ▼
  Baseline Model   Random Forest
  (Linear / Logistic)  (100 trees)
       │               │
       └───────┬───────┘
               ▼
  ┌─────────────────────────┐
  │   Metric Evaluation     │  ← R², MAE, RMSE / Accuracy, F1, AUC
  │   + Cross Validation    │    5-fold CV scores
  └────────────┬────────────┘
               │
               ▼
  ┌─────────────────────────┐
  │  Health Score (0–100)   │  ← composite quality signal
  └────────────┬────────────┘
               │
               ▼
  ┌──────────────────────────┐
  │  LLM Analysis Layer      │  ← Claude (Anthropic) → expert bullets
  │  (rule-based fallback)   │    references actual computed numbers
  └────────────┬─────────────┘
               │
               ▼
      Streamlit Dashboard
  (Metrics · Analysis · Features · Data Quality)
```

---

## 🧪 Running Tests

```bash
python test_engine.py
```

The test suite covers:

| # | Test Case | Checks |
|---|---|---|
| 1 | Regression — clean dataset | R², MAE, RMSE, CV |
| 2 | Classification — binary (breast cancer) | Accuracy, F1, AUC |
| 3 | Heavy missing values (30%) | Missing detection, imputation |
| 4 | Mixed numeric + categorical | Encoding, feature importance |
| 5 | Dataset too small (< 10 rows) | Guard clause fires correctly |
| 6 | No target column specified | Auto-detection of last column |

---



## ⚙️ Configuration Reference

| Setting | Where | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | `.env` or Streamlit secrets | Enables Claude LLM analysis |
| Fallback dataset | `data/initial_dataset.csv` | Auto-loaded when no CSV is uploaded |
| Target column | UI dropdown | Defaults to the last column |

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| UI | Streamlit 1.35+ |
| Visualisation | Plotly Express + Graph Objects |
| ML Pipeline | scikit-learn (Pipeline, ColumnTransformer) |
| LLM | Anthropic Claude (`claude-sonnet-4-5`) |
| Data | Pandas, NumPy |
| Testing | Pure Python (no pytest dependency) |

---

## 🔮 Roadmap

- [ ] SHAP values for model explainability
- [ ] PDF / HTML diagnostic report export
- [ ] Time-series dataset detection and stationarity tests
- [ ] Support for OpenAI GPT-4o as an alternative LLM backend
- [ ] Automated feature engineering suggestions

---



## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 👤 Author

**Nishant Diwate**  
[GitHub](https://github.com/nishantdiwate) · [Live App](https://jzpahqig75a375i8ycpiy2.streamlit.app/)

---

<p align="center">
  If this project helped you, consider giving it a ⭐ — it helps others find it.
</p>