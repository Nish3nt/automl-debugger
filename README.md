# 🧠 AutoML Debugger  
### LLM-Assisted Dataset Diagnostics for Machine Learning Engineers

> **Know whether your dataset is worth training on — before you waste compute.**

AutoML Debugger is an **industry-grade ML diagnostics tool** that automatically evaluates dataset quality, detects weak predictive signals, and explains results using **LLM-powered expert reasoning** — even when **no dataset is uploaded**.

This project is inspired by **real internal tools used by ML teams** to validate data *before* model deployment.

---

## ⭐ Why This Repo Deserves a Star

✔️ Solves a **real ML engineering problem**  
✔️ Handles **messy real-world datasets**  
✔️ Uses **LLMs for reasoning, not gimmicks**  
✔️ Robust fallback system (never breaks)  
✔️ Clean, recruiter-friendly UI  
✔️ Not another “train a model” project  

If you work with ML data — this tool is useful.

---

## 🚀 What Problem Does This Solve?

Most ML failures happen **because of bad data**, not bad models.

AutoML Debugger answers:
- ❓ *Does my dataset have predictive signal?*
- ❓ *Is this dataset production-ready?*
- ❓ *Why is my model performing poorly?*
- ❓ *What should I fix first?*

All **automatically**, with **clear explanations**.

---

## ✨ Key Features

### 📊 Quantitative ML Health Metrics
- Train / validation performance (R², error trends)
- Automatic weak-signal detection
- Clear dataset diagnosis

### 🧪 Robust Data Handling (Zero-Crash Design)
- Handles:
  - Missing values (NaNs)
  - Mixed numeric & categorical features
  - Invalid data types
- Uses sklearn Pipelines (production-safe)

### 🔁 Fallback Dataset Support (Important)
- If **no dataset is uploaded**:
  - The app automatically loads a **stored fallback dataset**
  - Runs the full AutoML debugging pipeline
- The **Run AutoML Debugger** button always works

### 🧠 LLM-Based Expert Analysis
- Uses a real LLM (OpenAI / Gemini-ready)
- Outputs **clear bullet-point explanations**, such as:
  - What the model learned
  - Why performance is weak or strong
  - Which dataset properties caused issues
- Written so **recruiters & non-ML stakeholders** understand

### 🖥️ Clean Streamlit UI
- Dark-mode friendly interface
- Step-by-step flow
- No disruptive red error banners
- Clear separation of metrics & explanations

---

## 🏗️ How It Works (System Overview)

1️⃣ **Dataset Ingestion**  
- User uploads CSV **or**
- App automatically loads fallback dataset  

2️⃣ **Preprocessing Pipeline**  
- Numeric → imputation + scaling  
- Categorical → safe encoding  
- Fully sklearn-pipeline based  

3️⃣ **Baseline Model Training**  
- Lightweight, interpretable regression model  
- Designed for diagnostics (not leaderboard chasing)

4️⃣ **Metric Evaluation**  
- Model performance computed
- Predictive signal strength assessed

5️⃣ **LLM Explanation Layer**  
- Metrics passed to LLM
- LLM returns **structured bullet-point analysis**
- Explains *why* results look the way they do

---

## 🧠 Example LLM Output

- The model shows **low R²**, indicating weak correlation between features and target.
- High noise and limited feature relevance reduce predictive power.
- Dataset likely requires:
  - Feature engineering
  - Target redefinition
  - Larger or cleaner data
- Current dataset is **not production-ready** without improvements.

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** — UI
- **Pandas / NumPy** — Data handling
- **Scikit-Learn (locked version)** — ML pipelines
- **LLM API (OpenAI / Gemini compatible)** — Expert reasoning

---

## 📦 Dependency Stability (No Version Chaos)

This project uses **locked versions** to guarantee reproducibility:

```txt
streamlit==1.31.0
pandas==2.1.4
numpy==1.26.4
scikit-learn==1.3.2
scipy==1.11.4
matplotlib==3.8.2


## Author
Nishant Diwate
