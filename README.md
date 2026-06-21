# 🧠 AutoML Debugger v2.0
### LLM-Assisted Dataset Diagnostics for Machine Learning Engineers

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white"/>
  <img src="https://img.shields.io/badge/Streamlit-1.35%2B-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white"/>
  <img src="https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white"/>
  <img src="https://img.shields.io/badge/Groq-LLaMA_3.3_70B-5A4FCF?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge"/>
</p>

<p align="center"><b>Know whether your dataset is worth training on — before you waste compute.</b></p>

---

## 🆕 v2.0 Advancements

| # | Feature | Description |
|---|---------|-------------|
| 1 | **Groq LLM** | Replaced Anthropic with Groq LLaMA 3.3 70B — faster, free tier |
| 2 | **Dataset Cleaning + Export** | One-click clean + download CSV after diagnosis |
| 3 | **Data Leakage Detection** | Flags features with >0.95 correlation with target |
| 4 | **Time-Series Detection** | Auto-detects datetime columns, uses chronological split |
| 5 | **PDF Report Export** | Full downloadable diagnostic report |

---

## ✨ Features

- Auto task detection (regression / classification)
- Real LLM analysis via Groq LLaMA 3.3 70B (rule-based fallback if no key)
- Dual-model evaluation (Linear/Logistic + Random Forest)
- 5-fold cross-validation (TimeSeriesSplit for time-series data)
- Rich data profiling: missing values, duplicates, outliers, imbalance, cardinality
- Data leakage detection with correlation table
- Time-series detection with frequency estimation
- One-click dataset cleaning (impute, cap, deduplicate, drop constants)
- PDF diagnostic report download
- Dataset Health Score (0–100)
- Interactive Plotly charts



---

## 📁 Project Structure

```
automl-debugger/
├── app.py                        # Streamlit UI
├── src/
│   ├── __init__.py
│   └── debugger_engine.py        # Full ML pipeline
├── data/
│   └── initial_dataset.csv       # Fallback sample dataset
├── .streamlit/
│   └── secrets.toml.example      # API key template
├── requirements.txt
├── .gitignore
└── README.md
```

---

## ⚙️ Configuration

| Setting | Where | Description |
|---------|-------|-------------|
| `GROQ_API_KEY` | `.streamlit/secrets.toml` or Streamlit Cloud Secrets | Enables Groq LLM analysis |
| Fallback dataset | `data/initial_dataset.csv` | Auto-loaded when no CSV uploaded |
| Target column | UI dropdown | Defaults to last column |
| Cleaning options | Sidebar toggles | Control what gets cleaned |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit 1.35+ |
| Visualisation | Plotly Express + Graph Objects |
| ML Pipeline | scikit-learn |
| LLM | Groq (LLaMA 3.3 70B) |
| PDF Generation | ReportLab |
| Data | Pandas, NumPy |

---

## 👤 Author

**Nishant Diwate** · [GitHub](https://github.com/nishantdiwate)
