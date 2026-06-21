"""
AutoML Debugger — Streamlit App  (Stage 2 v3.0)
================================================
Step-by-step wizard UI:
  Step 1  Upload & Preview
  Step 2  Confirm target + task type (with explanation)
  Step 3  Preprocessing summary (what happens & why)
  Step 4  Live training progress feed
  Step 5  Results — Overview, Model Leaderboard, Features,
                    Data Quality, Leakage, SHAP, Residuals/Confusion,
                    Feature Engineering, Drift, Model Card, Export
"""

import os
import time
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

st.set_page_config(
    page_title="AutoML Debugger",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.debugger_engine import (
    detect_task_type, detect_timeseries, detect_leakage,
    profile_dataset, build_preprocessing_summary,
    compute_health_score, clean_dataset,
    generate_llm_analysis, generate_pdf_report,
)
from src.model_selector import run_model_pipeline
from src.predictor import predict_from_dataframe, predict_single_row
from src.feature_engineer import suggest_feature_engineering
from src.explainer import (
    compute_shap_values, compute_residual_analysis,
    compute_confusion_analysis, detect_drift,
    generate_model_card, save_model_to_bytes,
)

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color: #0e1117; }
[data-testid="stSidebar"]  { background-color: #161b27; border-right: 1px solid #2a2f45; }
h1,h2,h3 { font-family:'Segoe UI',sans-serif; }

.kpi-card {
    background: linear-gradient(135deg,#1a1f35,#1e2540);
    border:1px solid #2e3555; border-radius:12px;
    padding:16px 20px; text-align:center;
    box-shadow:0 4px 15px rgba(0,0,0,.3);
    margin-bottom:8px;
}
.kpi-val   { font-size:1.8rem; font-weight:700; color:#7eb6ff; }
.kpi-label { font-size:0.75rem; color:#8892a4; text-transform:uppercase; letter-spacing:1px; margin-top:4px; }

.step-header {
    background:linear-gradient(90deg,#1a1f35,#1e2540);
    border-left:4px solid #7eb6ff; border-radius:8px;
    padding:12px 18px; margin-bottom:16px;
    font-size:1.1rem; font-weight:600; color:#e0e0e0;
}
.verdict-card {
    border-radius:12px; padding:20px 24px;
    margin-bottom:16px; font-size:1rem; line-height:1.6;
}
.verdict-go    { background:#1a2e1a; border:1px solid #5dbc8a; }
.verdict-fix   { background:#2e2a1a; border:1px solid #e0b070; }
.verdict-stop  { background:#2e1a1a; border:1px solid #e07070; }

.bullet-card {
    background:#161b27; border-left:3px solid #7eb6ff;
    border-radius:6px; padding:11px 15px;
    margin-bottom:9px; font-size:0.93rem; line-height:1.5;
}
.bullet-warn  { border-left-color:#e07070; background:#1e1515; }
.bullet-green { border-left-color:#5dbc8a; background:#151e15; }
.bullet-amber { border-left-color:#e0b070; background:#1e1b15; }

.prep-row {
    display:flex; align-items:flex-start; gap:12px;
    background:#161b27; border-radius:8px;
    padding:10px 14px; margin-bottom:7px;
}
.prep-badge {
    display:inline-block; padding:2px 10px; border-radius:12px;
    font-size:0.75rem; font-weight:600; white-space:nowrap; margin-top:2px;
}
.badge-impute  { background:#1a3a5c; color:#7eb6ff; }
.badge-encode  { background:#2a1a4c; color:#b07eff; }
.badge-drop    { background:#3a1a1a; color:#e07070; }
.badge-scale   { background:#1a3a2c; color:#5dbc8a; }
.badge-ts      { background:#2a2a1a; color:#e0e070; }

.progress-line {
    background:#161b27; border-radius:6px;
    padding:8px 14px; margin-bottom:5px;
    font-family:monospace; font-size:0.88rem; color:#7eb6ff;
}

.health-bar-bg {
    background:#1a1f35; border-radius:8px;
    height:18px; overflow:hidden; border:1px solid #2e3555;
}
.pill {
    display:inline-block; padding:3px 12px; border-radius:20px;
    font-size:0.78rem; font-weight:600; margin:3px;
}
.pill-blue  { background:#1a3a5c; color:#7eb6ff; border:1px solid #2a5080; }
.pill-green { background:#1a3a2c; color:#5dbc8a; border:1px solid #2a6040; }
.pill-red   { background:#3a1a1a; color:#e07070; border:1px solid #804040; }
.pill-amber { background:#3a2a1a; color:#e0b070; border:1px solid #806040; }

.model-rank-1 { background:linear-gradient(135deg,#2a2a0a,#2e2e12); border:1px solid #d4af37; border-radius:10px; padding:14px 18px; margin-bottom:10px; }
.model-rank-n { background:#161b27; border:1px solid #2e3555; border-radius:10px; padding:14px 18px; margin-bottom:10px; }

.dim-row {
    display:flex; justify-content:space-between; align-items:center;
    background:#161b27; border-radius:6px; padding:9px 14px; margin-bottom:6px;
}
.dim-label { font-size:0.9rem; color:#c0c8d8; font-weight:500; }
.dim-score { font-size:1rem; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def kpi(col, value, label):
    col.markdown(
        f'<div class="kpi-card"><div class="kpi-val">{value}</div>'
        f'<div class="kpi-label">{label}</div></div>',
        unsafe_allow_html=True,
    )

def bullet(text, style=""):
    cls = f"bullet-card {style}"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

def step_header(text):
    st.markdown(f'<div class="step-header">{text}</div>', unsafe_allow_html=True)

BADGE_MAP = {"imputation": "badge-impute", "encoding": "badge-encode",
             "drop": "badge-drop", "scaling": "badge-scale", "time-series": "badge-ts"}
BADGE_LABEL = {"imputation": "IMPUTE", "encoding": "ENCODE",
               "drop": "DROP", "scaling": "SCALE", "time-series": "TIME-SERIES"}

def prep_row(action, column, reason, category):
    badge_cls = BADGE_MAP.get(category, "badge-scale")
    badge_lbl = BADGE_LABEL.get(category, category.upper())
    st.markdown(f"""
    <div class="prep-row">
      <span class="prep-badge {badge_cls}">{badge_lbl}</span>
      <div>
        <strong style="color:#c0c8d8">{column}</strong>
        <span style="color:#8892a4; margin:0 6px">→</span>
        <span style="color:#7eb6ff">{action}</span><br>
        <span style="color:#6a7485;font-size:0.82rem">{reason}</span>
      </div>
    </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    api_key = ""
    if hasattr(st, "secrets"):
        api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input("🔑 Groq API Key (optional)", type="password",
            help="Enables Groq LLaMA 3.3 70B analysis. Leave blank for rule-based fallback.")

    st.divider()
    st.markdown("### 🧹 Cleaning Options")
    opt_dup     = st.checkbox("Remove duplicates",     value=True)
    opt_impute  = st.checkbox("Impute missing values", value=True)
    opt_out     = st.checkbox("Cap outliers (IQR)",    value=True)
    opt_const   = st.checkbox("Drop constant columns", value=True)

    st.divider()
    st.markdown("""### 📌 What This App Does
Upload any CSV dataset. The app will:
1. **Understand** your data automatically
2. **Explain** every preprocessing decision
3. **Train** Ridge + XGBoost + LightGBM
4. **Compare** models on a leaderboard
5. **Clean** and let you download your data
6. **Predict** on new inputs
7. **Export** a full PDF report
    """)
    st.divider()
    st.markdown("Built by [Nishant Diwate](https://github.com/nishantdiwate)")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
c1, c2 = st.columns([1, 10])
with c1:
    st.markdown("<h1 style='font-size:2.8rem;margin:0'>🧠</h1>", unsafe_allow_html=True)
with c2:
    st.markdown("<h1 style='margin:0;padding-top:8px'>AutoML Debugger</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8892a4;margin:0'>Upload your dataset → Get a trained model, expert analysis, and predictions in minutes</p>", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# STEP 1 — UPLOAD
# ─────────────────────────────────────────────
step_header("Step 1 — Upload Your Dataset")

FALLBACK = Path("data/initial_dataset.csv")
uploaded = st.file_uploader("Upload a CSV file", type=["csv"], label_visibility="collapsed")

df = None
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        if df.shape[0] < 10 or df.shape[1] < 2:
            st.error("❌ File too small — need at least 10 rows and 2 columns.")
            df = None
        elif df.shape[0] > 500_000:
            st.warning("⚠️ Large file — sampling 500,000 rows.")
            df = df.sample(500_000, random_state=42)
        if df is not None:
            st.success(f"✅ Uploaded — {df.shape[0]:,} rows × {df.shape[1]} columns")
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
elif FALLBACK.exists():
    df = pd.read_csv(FALLBACK)
    st.info(f"ℹ️ Using built-in sample dataset — {df.shape[0]:,} rows × {df.shape[1]} columns")

if df is None:
    st.warning("Please upload a CSV to continue.")
    st.stop()

with st.expander("🔍 Preview Dataset", expanded=False):
    st.dataframe(df.head(10), use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows",    f"{df.shape[0]:,}")
    c2.metric("Columns", str(df.shape[1]))
    c3.metric("Missing", f"{df.isna().sum().sum():,}")

st.divider()

# ─────────────────────────────────────────────
# STEP 2 — CONFIRM TARGET + TASK TYPE
# ─────────────────────────────────────────────
step_header("Step 2 — Confirm Target Column & Task Type")

col_left, col_right = st.columns([1, 1])

with col_left:
    no_target = st.checkbox("No target column (run clustering instead)", value=False)

    target_column = None
    if not no_target:
        target_column = st.selectbox(
            "Select target column (what to predict)",
            options=df.columns.tolist(),
            index=len(df.columns) - 1,
        )

with col_right:
    if no_target:
        st.info("🔵 **Mode: Clustering**\nKMeans will automatically find natural groups in your data (k=2 to 8). Best for exploration when you don't have a label.")
        task_type        = "clustering"
        task_reason      = "No target column selected — running unsupervised KMeans clustering."
        task_type_locked = True
    else:
        auto_type, auto_reason = detect_task_type(df[target_column])
        task_type_locked = False

        task_type = st.radio(
            "Task type",
            options=["regression", "classification"],
            index=0 if auto_type == "regression" else 1,
            horizontal=True,
        )

        reason_color = "#5dbc8a" if task_type == auto_type else "#e0b070"
        st.markdown(
            f'<div style="background:#161b27;border-left:3px solid {reason_color};'
            f'border-radius:6px;padding:10px 14px;font-size:0.88rem;color:#c0c8d8;margin-top:8px">'
            f'<b>Auto-detection:</b> {auto_reason}</div>',
            unsafe_allow_html=True,
        )
        if task_type != auto_type:
            st.warning(f"⚠️ You overrode auto-detection from '{auto_type}' to '{task_type}'. Make sure this is intentional.")

st.divider()

# ─────────────────────────────────────────────
# STEP 3 — PREPROCESSING SUMMARY
# ─────────────────────────────────────────────
step_header("Step 3 — Preprocessing Plan (What Happens Before Training & Why)")

ts_info = detect_timeseries(df)

if task_type != "clustering" and target_column:
    prep_steps = build_preprocessing_summary(df, target_column, ts_info.get("datetime_column"))

    if ts_info["is_timeseries"]:
        for w in ts_info["warnings"]:
            st.markdown(f'<div class="bullet-card bullet-amber">{w}</div>', unsafe_allow_html=True)

    cat_counts = {"imputation": 0, "encoding": 0, "drop": 0, "scaling": 0, "time-series": 0}
    for step in prep_steps:
        cat_counts[step["category"]] = cat_counts.get(step["category"], 0) + 1
        prep_row(step["action"], step["column"], step["reason"], step["category"])

    st.caption(
        f"Total: {cat_counts['imputation']} imputation(s) · "
        f"{cat_counts['encoding']} encoding(s) · "
        f"{cat_counts['drop']} drop(s) · "
        f"{cat_counts['scaling']} scaling step(s)"
    )
else:
    st.info("🔵 Clustering mode: numeric features will be median-imputed and StandardScaled. Categorical columns are excluded.")

st.divider()

# ─────────────────────────────────────────────
# RUN BUTTON
# ─────────────────────────────────────────────
run_btn = st.button("🚀 Run Full AutoML Pipeline", type="primary", use_container_width=True)

if not run_btn:
    st.stop()

# ─────────────────────────────────────────────
# STEP 4 — LIVE PROGRESS FEED
# ─────────────────────────────────────────────
st.divider()
step_header("Step 4 — Training Pipeline (Live)")

progress_placeholder = st.empty()
log_lines: list[str] = []

def progress_callback(msg: str):
    log_lines.append(msg)
    html = "".join(f'<div class="progress-line">{ln}</div>' for ln in log_lines[-20:])
    progress_placeholder.markdown(html, unsafe_allow_html=True)
    time.sleep(0.05)

progress_callback("🚀 AutoML pipeline started...")

# Profile & leakage (fast)
if task_type != "clustering" and target_column:
    progress_callback("🔍 Profiling dataset...")
    profile = profile_dataset(df, target_column)
    progress_callback(
        f"✅ Profile complete — {profile['rows']:,} rows, "
        f"{profile['missing_pct']}% missing, "
        f"{profile['duplicate_rows']} duplicates"
    )

    progress_callback("🚨 Running data leakage detection...")
    leakage = detect_leakage(df, target_column)
    if leakage["leakage_candidates"]:
        progress_callback(f"🚨 Leakage risk: {leakage['leakage_candidates']}")
    else:
        progress_callback("✅ No leakage features detected.")
else:
    profile = {}
    leakage = {}

# Cleaning
progress_callback("🧹 Cleaning dataset...")
if task_type != "clustering" and target_column:
    cleaned_df, cleaning_report = clean_dataset(
        df, target_column,
        remove_duplicates=opt_dup,
        impute_missing=opt_impute,
        cap_outliers=opt_out,
        drop_constant=opt_const,
    )
    for action in cleaning_report.get("actions", []):
        progress_callback(f"  ✅ {action}")
else:
    cleaned_df, cleaning_report = df.copy(), {}

# Model training
progress_callback("🤖 Starting model training...")
model_results = run_model_pipeline(
    df=cleaned_df if task_type != "clustering" else df,
    target_column=target_column,
    task_type=task_type,
    is_timeseries=ts_info.get("is_timeseries", False),
    ts_datetime_col=ts_info.get("datetime_column"),
    progress_callback=progress_callback,
)

feature_importance = model_results.get("feature_importance", {})
leaderboard        = model_results.get("leaderboard", [])
best_model         = leaderboard[0] if leaderboard else {}

# Health score
best_metrics_for_health = {}
if task_type == "regression":
    best_metrics_for_health = {
        "best_r2": best_model.get("cv_score", 0),
        "r2":      best_model.get("test_r2", 0),
    }
else:
    best_metrics_for_health = {
        "best_accuracy": best_model.get("cv_score", 0),
        "accuracy":      best_model.get("test_accuracy", 0),
    }

if task_type != "clustering":
    health = compute_health_score(profile, best_metrics_for_health, task_type, leakage)
else:
    health = {"total": 0, "verdict": "N/A (clustering)", "dimensions": {}}

# LLM analysis
progress_callback("🤖 Generating expert analysis (Groq LLaMA 3.3 70B)..." if api_key else "⚙️ Generating rule-based analysis...")
llm_analysis = generate_llm_analysis(
    profile, model_results, task_type, health,
    leakage, ts_info, feature_importance,
    prep_steps if task_type != "clustering" else [],
    api_key=api_key or None,
) if task_type != "clustering" else {}

# PDF
progress_callback("📄 Generating PDF report...")
pdf_bytes = None
if task_type != "clustering":
    pdf_bytes = generate_pdf_report(
        profile, model_results, task_type, health,
        llm_analysis, feature_importance, leakage, ts_info,
        cleaning_report,
    )

progress_callback("✅ All done!")

# ─────────────────────────────────────────────
# STEP 5 — RESULTS
# ─────────────────────────────────────────────
st.divider()
step_header("Step 5 — Results")

# ── CLUSTERING RESULTS ───────────────────────────────────────────
if task_type == "clustering":
    st.subheader("🔵 Clustering Results")
    best_k  = model_results.get("best_k", "?")
    sizes   = model_results.get("cluster_sizes", {})
    inertia = model_results.get("inertia", 0)

    c1, c2, c3 = st.columns(3)
    c1.metric("Optimal Clusters (k)", best_k)
    c2.metric("Total Inertia", f"{inertia:.1f}")
    c3.metric("Largest Cluster", max(sizes.values()) if sizes else "?")

    st.markdown("#### Cluster Sizes")
    size_df = pd.DataFrame(list(sizes.items()), columns=["Cluster", "Size"])
    fig = px.bar(size_df, x="Cluster", y="Size", color="Cluster",
                 template="plotly_dark", color_discrete_sequence=px.colors.sequential.Blues_r)
    fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    elbow = model_results.get("elbow_data", {})
    if elbow:
        fig2 = go.Figure(go.Scatter(
            x=elbow["k_values"], y=elbow["inertias"],
            mode="lines+markers", line=dict(color="#7eb6ff", width=2),
            marker=dict(size=8, color="#7eb6ff"),
        ))
        fig2.update_layout(
            title="Elbow Method — Inertia vs K",
            xaxis_title="Number of Clusters (k)", yaxis_title="Inertia",
            template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### Cluster Profiles (Mean Feature Values)")
    profiles = model_results.get("profiles")
    if profiles is not None:
        st.dataframe(profiles, use_container_width=True, hide_index=True)

    st.stop()

# ── SUPERVISED RESULTS ───────────────────────────────────────────

# Verdict card
total = health["total"]
verdict = health["verdict"]
if total >= 80:
    vclass = "verdict-go"
    vicon  = "🟢"
    vmsg   = "Your dataset is production-ready. Models show strong signal — proceed to full model training."
elif total >= 60:
    vclass = "verdict-fix"
    vicon  = "🟡"
    vmsg   = "Dataset is usable but needs minor work. Review the action plan below before deploying."
else:
    vclass = "verdict-stop"
    vicon  = "🔴"
    vmsg   = "Significant data quality issues detected. Fix critical items before training any production model."

st.markdown(
    f'<div class="verdict-card {vclass}">'
    f'{vicon} <b>Verdict: {verdict}</b> — Health Score {total}/100<br>'
    f'<span style="color:#c0c8d8">{vmsg}</span></div>',
    unsafe_allow_html=True,
)

# Banners
if ts_info.get("is_timeseries"):
    st.markdown(
        f'<div class="bullet-card bullet-amber">⏱️ <b>Time-Series Detected</b> — '
        f'column <code>{ts_info["datetime_column"]}</code>, frequency: {ts_info["frequency_guess"]}. '
        f'Chronological split applied.</div>', unsafe_allow_html=True,
    )
if leakage.get("leakage_candidates"):
    st.markdown(
        f'<div class="bullet-card bullet-warn">🚨 <b>Data Leakage Risk</b> — '
        f'{len(leakage["leakage_candidates"])} feature(s) with >0.95 correlation with target: '
        f'<code>{", ".join(leakage["leakage_candidates"])}</code></div>', unsafe_allow_html=True,
    )

# Top KPIs
st.markdown("<br>", unsafe_allow_html=True)
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi(k1, f"{profile.get('rows',0):,}", "Rows")
kpi(k2, str(profile.get('columns',0)), "Columns")
kpi(k3, f"{profile.get('missing_pct',0)}%", "Missing")
kpi(k4, str(profile.get('duplicate_rows',0)), "Duplicates")
kpi(k5, task_type.capitalize(), "Task Type")
kpi(k6, f"{total}/100", "Health Score")

st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────
# ── Stage 2 computations ─────────────────────────────────────
feature_suggestions = []
shap_data           = {}
residual_analysis   = {}
confusion_analysis  = {}
model_card_md       = ""
model_bytes         = None

if task_type != "clustering" and best_pipeline:
    # Feature engineering suggestions
    feature_suggestions = suggest_feature_engineering(
        cleaned_df, target_column, feature_importance, profile, task_type,
    )

    # SHAP values
    with st.spinner("Computing SHAP explanations..."):
        try:
            nf2 = model_results.get("numeric_features", [])
            cf2 = model_results.get("categorical_features", [])
            X_for_shap = cleaned_df.drop(columns=[target_column]).head(300)
            shap_data = compute_shap_values(best_pipeline, X_for_shap, nf2, cf2)
        except Exception:
            shap_data = {"available": False, "error": "SHAP computation failed"}

    # Residual / confusion analysis
    X_test_s  = model_results.get("X_test")
    y_test_s  = model_results.get("y_test")
    le_s      = model_results.get("label_encoder")
    if X_test_s is not None and y_test_s is not None and best_pipeline is not None:
        y_pred_s = best_pipeline.predict(X_test_s)
        if task_type == "regression":
            residual_analysis = compute_residual_analysis(y_test_s, y_pred_s, best_model.get("model",""))
        else:
            confusion_analysis = compute_confusion_analysis(y_test_s, y_pred_s, le_s, best_model.get("model",""))

    # Model card
    model_card_md = generate_model_card(
        model_name=best_model.get("model", "Unknown"),
        task_type=task_type,
        target_column=target_column,
        profile=profile,
        health=health,
        leaderboard=leaderboard,
        feature_importance=feature_importance,
        leakage=leakage,
        ts_info=ts_info,
        cleaning_report=cleaning_report,
    )

    # Save model bytes
    try:
        model_bytes = save_model_to_bytes(
            best_pipeline, le_s,
            model_results.get("numeric_features", []),
            model_results.get("categorical_features", []),
            task_type, best_model.get("model", "Model"), target_column,
        )
    except Exception:
        model_bytes = None

(tab_overview, tab_models, tab_features, tab_quality, tab_leakage,
 tab_shap, tab_eval, tab_fe, tab_drift, tab_predict, tab_export) = st.tabs([
    "📊 Overview",
    "🏆 Model Leaderboard",
    "🌲 Feature Intelligence",
    "🔍 Data Quality",
    "🚨 Leakage & Time-Series",
    "🔮 SHAP Explainability",
    "📉 Residuals / Confusion",
    "⚙️ Feature Engineering",
    "📡 Drift Detection",
    "🎯 Predictions",
    "📄 Export",
])

# ── TAB 1: OVERVIEW ──────────────────────────
with tab_overview:
    # Health score breakdown
    st.subheader(f"⭐ Dataset Health Score — {verdict}")
    bar_col = "#5dbc8a" if total >= 80 else ("#e0b070" if total >= 60 else "#e07070")
    st.markdown(
        f'<div class="health-bar-bg"><div style="width:{total}%;background:{bar_col};'
        f'height:100%;border-radius:8px;"></div></div>'
        f'<p style="color:{bar_col};font-size:1.5rem;font-weight:700;margin:6px 0 16px">{total} / 100</p>',
        unsafe_allow_html=True,
    )

    st.markdown("**Score Breakdown (hover for details)**")
    for dim_name, dim_data in health["dimensions"].items():
        pct  = int(dim_data["score"] / dim_data["max"] * 100)
        col  = "#5dbc8a" if pct >= 80 else ("#e0b070" if pct >= 50 else "#e07070")
        st.markdown(
            f'<div class="dim-row">'
            f'<span class="dim-label">{dim_name}</span>'
            f'<span style="flex:1;margin:0 14px"><div class="health-bar-bg" style="height:10px">'
            f'<div style="width:{pct}%;background:{col};height:100%;border-radius:8px;"></div></div></span>'
            f'<span class="dim-score" style="color:{col}">{dim_data["score"]}/{dim_data["max"]}</span>'
            f'</div>'
            f'<div style="color:#5a6275;font-size:0.8rem;margin:-4px 0 8px 8px">{dim_data["reason"]}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # Overall analysis
    st.subheader("🧠 Expert Analysis")
    if api_key:
        st.caption("Powered by Groq LLaMA 3.3 70B")
    else:
        st.caption("Rule-based analysis — add Groq API key in sidebar for LLM commentary")

    for b in llm_analysis.get("overall_analysis", []):
        bullet(b)

    st.divider()
    st.subheader("✅ Action Plan — Do These Before Training")
    for b in llm_analysis.get("action_plan", []):
        bullet(b, "bullet-amber")

# ── TAB 2: MODEL LEADERBOARD ─────────────────
with tab_models:
    st.subheader("🏆 Model Leaderboard")

    baseline = model_results.get("baseline", {})
    if baseline:
        st.markdown(
            f'<div class="bullet-card bullet-amber">📏 <b>Baseline ({baseline.get("model","?")}):</b> '
            f'{baseline.get("description","")} — '
            f'{"R²=" + str(baseline.get("r2","?")) if task_type=="regression" else "Accuracy=" + str(baseline.get("accuracy","?"))}'
            f' — Your model MUST beat this to be useful.</div>',
            unsafe_allow_html=True,
        )

    for i, model in enumerate(leaderboard):
        card_cls = "model-rank-1" if i == 0 else "model-rank-n"
        rank_badge = "🥇" if i == 0 else ("🥈" if i == 1 else "🥉")

        if task_type == "regression":
            metrics_html = (
                f"CV R²: <b>{model.get('cv_score','?')}</b> &nbsp;|&nbsp; "
                f"Train R²: <b>{model.get('train_score','?')}</b> &nbsp;|&nbsp; "
                f"Test R²: <b>{model.get('test_r2','?')}</b> &nbsp;|&nbsp; "
                f"MAE: <b>{model.get('mae','?')}</b> &nbsp;|&nbsp; "
                f"RMSE: <b>{model.get('rmse','?')}</b>"
            )
            gap = abs((model.get('train_score') or 0) - (model.get('cv_score') or 0))
            overfit_note = (
                f' &nbsp;⚠️ Train/CV gap={gap:.3f} — possible overfit'
                if gap > 0.1 else
                f' &nbsp;✅ Train/CV gap={gap:.3f} — good generalization'
            )
        else:
            metrics_html = (
                f"CV Acc: <b>{model.get('cv_score','?')}</b> &nbsp;|&nbsp; "
                f"Train Acc: <b>{model.get('train_score','?')}</b> &nbsp;|&nbsp; "
                f"Test Acc: <b>{model.get('test_accuracy','?')}</b> &nbsp;|&nbsp; "
                f"F1: <b>{model.get('f1_score','?')}</b> &nbsp;|&nbsp; "
                f"AUC: <b>{model.get('roc_auc','?')}</b>"
            )
            gap = abs((model.get('train_score') or 0) - (model.get('cv_score') or 0))
            overfit_note = (
                f' &nbsp;⚠️ Train/CV gap={gap:.3f} — possible overfit'
                if gap > 0.1 else
                f' &nbsp;✅ Train/CV gap={gap:.3f} — good generalization'
            )

        st.markdown(
            f'<div class="{card_cls}">'
            f'<div style="font-size:1rem;font-weight:700;color:#e0e0e0;margin-bottom:6px">'
            f'{rank_badge} {model.get("model","?")}</div>'
            f'<div style="font-size:0.88rem;color:#8892a4">{metrics_html}</div>'
            f'<div style="font-size:0.8rem;color:#6a7485;margin-top:4px">{overfit_note}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.divider()
    st.subheader("💬 Model Commentary")
    for b in llm_analysis.get("model_commentary", []):
        bullet(b)

    # Tier 1 comparison chart
    if model_results.get("tier1_results"):
        st.divider()
        st.subheader("⚡ Tier 1 Screening Results (sample)")
        t1 = [r for r in model_results["tier1_results"] if r.get("cv_score", -999) > -999]
        if t1:
            t1_df = pd.DataFrame(t1)[["model","cv_score","cv_std"]].rename(
                columns={"cv_score": "CV Score", "cv_std": "CV Std", "model": "Model"}
            )
            fig = px.bar(t1_df.sort_values("CV Score"), x="CV Score", y="Model",
                         orientation="h", error_x="CV Std", template="plotly_dark",
                         color="CV Score", color_continuous_scale="Blues")
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
                              coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

# ── TAB 3: FEATURE INTELLIGENCE ──────────────
with tab_features:
    if feature_importance:
        st.subheader("🌲 Feature Importance (Best Model)")

        fi_df = pd.DataFrame({
            "Feature":    list(feature_importance.keys()),
            "Importance": list(feature_importance.values()),
        }).sort_values("Importance", ascending=True)

        fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale="Blues", template="plotly_dark")
        fig_fi.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
                             coloraxis_showscale=False,
                             xaxis=dict(gridcolor="#2e3555"), yaxis=dict(gridcolor="#2e3555"))
        st.plotly_chart(fig_fi, use_container_width=True)

        st.divider()
        st.subheader("💬 Feature Intelligence Commentary")
        for b in llm_analysis.get("feature_commentary", []):
            bullet(b)

        if profile.get("top_correlations"):
            st.divider()
            st.subheader("📈 Pearson Correlation with Target")
            corr_df = pd.DataFrame({
                "Feature":     list(profile["top_correlations"].keys()),
                "Correlation": list(profile["top_correlations"].values()),
            }).sort_values("Correlation", key=abs, ascending=True)

            fig_c = px.bar(corr_df, x="Correlation", y="Feature", orientation="h",
                           color="Correlation", color_continuous_scale="RdBu",
                           color_continuous_midpoint=0, template="plotly_dark")
            fig_c.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
                                coloraxis_showscale=False)
            st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("Feature importance not available.")

# ── TAB 4: DATA QUALITY ──────────────────────
with tab_quality:
    q1, q2 = st.columns(2)

    with q1:
        st.subheader("🧩 Missing Values")
        ms = df.isna().sum()
        ms = ms[ms > 0]
        if not ms.empty:
            miss_df = ms.reset_index()
            miss_df.columns = ["Column", "Missing"]
            fig_m = px.bar(miss_df, x="Missing", y="Column", orientation="h",
                           color="Missing", color_continuous_scale="Reds", template="plotly_dark")
            fig_m.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
                                coloraxis_showscale=False)
            st.plotly_chart(fig_m, use_container_width=True)
        else:
            st.success("✅ No missing values!")

    with q2:
        st.subheader("📊 Target Distribution")
        if task_type == "classification":
            vc = df[target_column].value_counts().reset_index()
            vc.columns = ["Class", "Count"]
            fig_d = px.pie(vc, names="Class", values="Count", template="plotly_dark",
                           color_discrete_sequence=px.colors.sequential.Blues_r)
        else:
            fig_d = px.histogram(df, x=target_column, template="plotly_dark",
                                 color_discrete_sequence=["#7eb6ff"], nbins=40)
        fig_d.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35")
        st.plotly_chart(fig_d, use_container_width=True)

    if profile.get("outlier_counts"):
        st.subheader("⚠️ Outlier Summary")
        out_df = pd.DataFrame({
            "Feature": list(profile["outlier_counts"].keys()),
            "Outlier Count": list(profile["outlier_counts"].values()),
        }).sort_values("Outlier Count", ascending=False)
        st.dataframe(out_df, use_container_width=True, hide_index=True)

    st.subheader("🏷️ Data Quality Flags")
    flags = []
    if profile.get("constant_features"):
        flags.append(f'<span class="pill pill-red">⚠️ {len(profile["constant_features"])} constant feature(s)</span>')
    if profile.get("high_cardinality_cols"):
        flags.append(f'<span class="pill pill-amber">⚠️ {len(profile["high_cardinality_cols"])} high-cardinality col(s)</span>')
    if profile.get("duplicate_rows", 0) > 0:
        flags.append(f'<span class="pill pill-amber">⚠️ {profile["duplicate_rows"]} duplicates</span>')
    if profile.get("missing_pct", 0) > 10:
        flags.append(f'<span class="pill pill-red">⚠️ {profile["missing_pct"]}% missing</span>')
    if profile.get("imbalance_ratio") and profile["imbalance_ratio"] > 3:
        flags.append(f'<span class="pill pill-amber">⚠️ Imbalance {profile["imbalance_ratio"]}:1</span>')
    if leakage.get("leakage_candidates"):
        flags.append(f'<span class="pill pill-red">🚨 {len(leakage["leakage_candidates"])} leakage risk(s)</span>')
    if ts_info.get("is_timeseries"):
        flags.append('<span class="pill pill-blue">⏱️ Time-series</span>')

    st.markdown(" ".join(flags) if flags else '<span class="pill pill-green">✅ No critical flags</span>',
                unsafe_allow_html=True)

# ── TAB 5: LEAKAGE & TIME-SERIES ─────────────
with tab_leakage:
    st.subheader("⏱️ Time-Series Analysis")
    if ts_info.get("is_timeseries"):
        c1, c2 = st.columns(2)
        c1.metric("DateTime Column",  ts_info.get("datetime_column", "—"))
        c2.metric("Frequency",        ts_info.get("frequency_guess", "—"))
        for w in ts_info.get("warnings", []):
            bullet(w, "bullet-amber")
    else:
        st.success("✅ No time-series structure detected — standard random split is safe.")

    st.divider()
    st.subheader("🚨 Data Leakage Detection")
    if leakage.get("leakage_candidates"):
        st.error(f"**{len(leakage['leakage_candidates'])} leakage candidate(s) found** — correlation > 0.95 with target.")
        for w in leakage.get("warnings", []):
            bullet(w, "bullet-warn")
    else:
        st.success("✅ No leakage features detected (no feature has correlation > 0.95 with target).")

    if leakage.get("high_correlation_features"):
        st.subheader("📊 All Feature Correlations with Target")
        corr_sorted = sorted(leakage["high_correlation_features"].items(),
                             key=lambda x: abs(x[1]), reverse=True)
        corr_df = pd.DataFrame(corr_sorted, columns=["Feature", "Abs Correlation"])
        corr_df["Risk"] = corr_df["Abs Correlation"].apply(
            lambda x: "🚨 HIGH RISK" if x > 0.95 else ("⚠️ Watch" if x > 0.85 else "✅ OK")
        )
        st.dataframe(corr_df, use_container_width=True, hide_index=True)

# ── TAB 6: PREDICTIONS ───────────────────────
with tab_predict:
    best_pipeline  = model_results.get("best_pipeline")
    label_encoder  = model_results.get("label_encoder")
    num_feats      = model_results.get("numeric_features", [])
    cat_feats      = model_results.get("categorical_features", [])
    pred_std       = best_model.get("pred_std", 0.0)

    if best_pipeline is None:
        st.warning("No trained model available for predictions.")
    else:
        st.subheader("🎯 Make Predictions")
        st.caption(f"Using: **{best_model.get('model','?')}** (best model from leaderboard)")

        pred_mode = st.radio("Prediction mode", ["Upload new CSV", "Fill a form"], horizontal=True)

        if pred_mode == "Upload new CSV":
            pred_file = st.file_uploader("Upload CSV without target column", type=["csv"],
                                         key="pred_upload")
            if pred_file:
                df_new = pd.read_csv(pred_file)
                st.dataframe(df_new.head(5), use_container_width=True)
                if st.button("Run Predictions", type="primary"):
                    result_df = predict_from_dataframe(
                        df_new, best_pipeline, label_encoder, task_type,
                        num_feats, cat_feats, pred_std,
                    )
                    st.success(f"✅ Predictions complete — {len(result_df):,} rows")
                    st.dataframe(result_df, use_container_width=True)
                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Predictions CSV", csv_out,
                                       "predictions.csv", "text/csv")

        else:  # Form mode
            st.markdown("Fill in the feature values:")
            all_feats = num_feats + cat_feats
            input_dict: dict = {}
            cols_per_row = 3
            rows = [all_feats[i:i+cols_per_row] for i in range(0, len(all_feats), cols_per_row)]
            for row_feats in rows:
                row_cols = st.columns(len(row_feats))
                for col_widget, feat in zip(row_cols, row_feats):
                    if feat in num_feats:
                        input_dict[feat] = col_widget.number_input(feat, value=0.0, key=f"inp_{feat}")
                    else:
                        unique_vals = df[feat].dropna().unique().tolist() if feat in df.columns else []
                        if unique_vals:
                            input_dict[feat] = col_widget.selectbox(feat, unique_vals, key=f"inp_{feat}")
                        else:
                            input_dict[feat] = col_widget.text_input(feat, key=f"inp_{feat}")

            if st.button("Predict", type="primary"):
                pred_out = predict_single_row(
                    input_dict, best_pipeline, label_encoder, task_type,
                    num_feats, cat_feats, pred_std,
                )
                st.success(f"**Prediction: {pred_out['prediction']}**")
                if "interval" in pred_out:
                    st.info(f"95% Confidence Interval: {pred_out['interval']}")
                if "confidence" in pred_out:
                    st.info(f"Prediction confidence: {pred_out['confidence']*100:.1f}%")
                if pred_out.get("probabilities"):
                    st.markdown("**Class probabilities:**")
                    for cls, prob in pred_out["probabilities"].items():
                        st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

# ── TAB 7: EXPORT ────────────────────────────
# ── TAB: SHAP EXPLAINABILITY ─────────────────────────────────
with tab_shap:
    st.subheader("🔮 SHAP Explainability")
    st.caption("SHAP (SHapley Additive exPlanations) shows how much each feature contributes to each prediction.")

    if not shap_data.get("available"):
        err = shap_data.get("error", "SHAP not available for this model.")
        st.warning(f"⚠️ {err}")
        st.info("SHAP works best with XGBoost and LightGBM. If using Ridge/Logistic, LinearExplainer is used.")
    else:
        # Global SHAP importance
        gi = shap_data.get("global_importance", {})
        if gi:
            st.markdown("#### 🌍 Global Feature Impact (Mean |SHAP|)")
            st.caption("Unlike feature importance (which is model-internal), SHAP values show the actual impact on predictions in the target's units.")
            gi_df = pd.DataFrame({"Feature": list(gi.keys()), "Mean |SHAP|": list(gi.values())}).sort_values("Mean |SHAP|", ascending=True)
            fig_shap = px.bar(gi_df, x="Mean |SHAP|", y="Feature", orientation="h",
                              color="Mean |SHAP|", color_continuous_scale="Blues", template="plotly_dark")
            fig_shap.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35", coloraxis_showscale=False)
            st.plotly_chart(fig_shap, use_container_width=True)

        # SHAP vs Feature Importance comparison
        if feature_importance and gi:
            st.markdown("#### 🔁 SHAP vs Feature Importance Comparison")
            st.caption("SHAP and feature importance often agree — but differences reveal features that rank high internally but have low real-world impact, or vice versa.")
            common = set(gi.keys()) & set(feature_importance.keys())
            if common:
                comp_df = pd.DataFrame([
                    {"Feature": f, "SHAP (normalised)": gi[f]/max(gi.values()), "FI (normalised)": feature_importance[f]/max(feature_importance.values())}
                    for f in list(common)[:10]
                ])
                fig_comp = px.scatter(comp_df, x="FI (normalised)", y="SHAP (normalised)", text="Feature",
                                      template="plotly_dark", color_discrete_sequence=["#7eb6ff"])
                fig_comp.update_traces(textposition="top center", textfont_size=9)
                fig_comp.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35")
                st.plotly_chart(fig_comp, use_container_width=True)

        # Per-sample explanations
        sample_exp = shap_data.get("sample_explanations", [])
        if sample_exp:
            st.markdown("#### 🔍 Per-Prediction Explanations (First 5 rows)")
            st.caption("For each prediction, SHAP shows which features pushed the result up (+) or down (−) from the baseline.")
            for s in sample_exp:
                with st.expander(f"Row {s['row']+1} — Top {len(s['top_features'])} drivers"):
                    for feat in s["top_features"]:
                        direction = "🔺 pushed prediction UP" if feat["shap"] > 0 else "🔻 pushed prediction DOWN"
                        color     = "#5dbc8a" if feat["shap"] > 0 else "#e07070"
                        st.markdown(
                            f'<div class="bullet-card" style="border-left-color:{color}">'
                            f'<b>{feat["feature"]}</b>: SHAP = {feat["shap"]:+.4f} — {direction}</div>',
                            unsafe_allow_html=True,
                        )

# ── TAB: RESIDUALS / CONFUSION ───────────────────────────────
with tab_eval:
    if task_type == "regression":
        st.subheader("📉 Residual Analysis")
        st.caption("Residuals = actual − predicted. Patterns in residuals reveal where and why the model fails.")

        if residual_analysis:
            stats = residual_analysis.get("stats", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Mean Residual",    stats.get("mean_residual", "—"))
            c2.metric("MAE",              stats.get("mae", "—"))
            c3.metric("Within 10% Error", f"{stats.get('within_10pct','?')}%")
            c4.metric("Within 20% Error", f"{stats.get('within_20pct','?')}%")

            for interp in residual_analysis.get("interpretations", []):
                style = "bullet-warn" if "⚠️" in interp or "🔴" in interp else ("bullet-green" if "✅" in interp else "")
                bullet(interp, style)

            plot_d = residual_analysis.get("plot_data", {})
            if plot_d:
                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown("**Actual vs Predicted**")
                    fig_avp = px.scatter(x=plot_d["y_true"], y=plot_d["y_pred"],
                                         labels={"x": "Actual", "y": "Predicted"},
                                         template="plotly_dark", color_discrete_sequence=["#7eb6ff"])
                    mn = min(min(plot_d["y_true"]), min(plot_d["y_pred"]))
                    mx = max(max(plot_d["y_true"]), max(plot_d["y_pred"]))
                    fig_avp.add_shape(type="line", x0=mn, x1=mx, y0=mn, y1=mx,
                                      line=dict(color="#e07070", dash="dash"))
                    fig_avp.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35")
                    st.plotly_chart(fig_avp, use_container_width=True)

                with col_b:
                    st.markdown("**Residuals vs Predicted**")
                    fig_res = px.scatter(x=plot_d["y_pred"], y=plot_d["residuals"],
                                          labels={"x": "Predicted", "y": "Residual"},
                                          template="plotly_dark", color_discrete_sequence=["#e0b070"])
                    fig_res.add_hline(y=0, line_color="#e07070", line_dash="dash")
                    fig_res.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35")
                    st.plotly_chart(fig_res, use_container_width=True)

                # Residual histogram
                st.markdown("**Residual Distribution**")
                fig_rh = px.histogram(x=plot_d["residuals"], nbins=40, template="plotly_dark",
                                       color_discrete_sequence=["#7eb6ff"],
                                       labels={"x": "Residual"})
                fig_rh.add_vline(x=0, line_color="#e07070", line_dash="dash")
                fig_rh.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35")
                st.plotly_chart(fig_rh, use_container_width=True)
        else:
            st.info("Residual analysis not available.")

    else:
        st.subheader("🔲 Confusion Matrix")
        st.caption("Shows where the model makes correct and incorrect predictions for each class.")

        if confusion_analysis:
            for interp in confusion_analysis.get("interpretations", []):
                style = "bullet-warn" if "🔴" in interp or "⚠️" in interp else ("bullet-green" if "✅" in interp else "")
                bullet(interp, style)

            cm_data   = confusion_analysis.get("confusion_matrix", [])
            classes_c = confusion_analysis.get("classes", [])
            if cm_data and classes_c:
                cm_arr = __import__("numpy").array(cm_data)
                fig_cm = px.imshow(cm_arr, x=classes_c, y=classes_c,
                                   color_continuous_scale="Blues", template="plotly_dark",
                                   labels={"x": "Predicted", "y": "Actual"},
                                   text_auto=True)
                fig_cm.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35")
                st.plotly_chart(fig_cm, use_container_width=True)

            # Per-class table
            pc = confusion_analysis.get("per_class", [])
            if pc:
                st.markdown("**Per-Class Metrics**")
                pc_df = pd.DataFrame(pc)
                st.dataframe(pc_df, use_container_width=True, hide_index=True)
        else:
            st.info("Confusion analysis not available.")

# ── TAB: FEATURE ENGINEERING ─────────────────────────────────
with tab_fe:
    st.subheader("⚙️ Feature Engineering Suggestions")
    st.caption("Concrete, code-ready transformations to improve model performance.")

    if not feature_suggestions:
        st.success("✅ No critical feature engineering needed — dataset looks well-structured.")
    else:
        priority_colors = {"HIGH": "#e07070", "MEDIUM": "#e0b070", "LOW": "#7eb6ff"}
        priority_bg     = {"HIGH": "#2e1a1a", "MEDIUM": "#2e2a1a", "LOW": "#1a2a3a"}

        for sug in feature_suggestions:
            pri   = sug.get("priority", "LOW")
            color = priority_colors.get(pri, "#7eb6ff")
            bg    = priority_bg.get(pri, "#161b27")
            st.markdown(
                f'<div style="background:{bg};border-left:3px solid {color};border-radius:8px;'
                f'padding:12px 16px;margin-bottom:10px">'
                f'<span style="color:{color};font-weight:700;font-size:0.78rem">{pri} PRIORITY</span> &nbsp;'
                f'<span style="color:#8892a4;font-size:0.78rem">{sug["type"]}</span><br>'
                f'<b style="color:#e0e0e0">{sug["suggestion"]}</b><br>'
                f'<span style="color:#8892a4;font-size:0.85rem">{sug["reason"]}</span><br>'
                f'<code style="background:#0e1117;padding:6px 10px;border-radius:4px;'
                f'font-size:0.82rem;color:#7eb6ff;display:block;margin-top:8px">{sug["code_snippet"]}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

# ── TAB: DRIFT DETECTION ─────────────────────────────────────
with tab_drift:
    st.subheader("📡 Train/Test Distribution Drift Detection")
    st.caption("Upload a separate test/production CSV to check if its distributions match your training data.")

    drift_file = st.file_uploader("Upload test/production dataset (CSV)", type=["csv"], key="drift_upload")

    if drift_file:
        df_test_drift = pd.read_csv(drift_file)
        st.success(f"✅ Test dataset loaded — {df_test_drift.shape[0]:,} rows × {df_test_drift.shape[1]} columns")

        if st.button("🔍 Run Drift Analysis", type="primary"):
            with st.spinner("Running distribution drift detection..."):
                drift_result = detect_drift(
                    cleaned_df, df_test_drift,
                    target_column=target_column if task_type != "clustering" else None,
                )

            # Verdict
            verdict_d = drift_result.get("verdict", "")
            vcolor = "#e07070" if "HIGH" in verdict_d else ("#e0b070" if "MODERATE" in verdict_d else "#5dbc8a")
            st.markdown(
                f'<div style="background:#1a1f35;border-left:4px solid {vcolor};border-radius:8px;'
                f'padding:14px 18px;margin-bottom:14px;font-size:1rem;color:#e0e0e0">{verdict_d}</div>',
                unsafe_allow_html=True,
            )

            for interp in drift_result.get("interpretations", []):
                bullet(interp)

            # Drift table
            dr = drift_result.get("drift_results", [])
            if dr:
                st.markdown("#### Per-Feature Drift Report")
                dr_display = []
                for r in dr:
                    row = {
                        "Feature":         r["feature"],
                        "Type":            r["type"],
                        "Drift Detected":  "🔴 YES" if r["drift_detected"] else "🟢 NO",
                        "Severity":        r.get("severity", ""),
                    }
                    if r["type"] == "numeric":
                        row["Train Mean"] = r.get("train_mean", "")
                        row["Test Mean"]  = r.get("test_mean", "")
                        row["Mean Shift"] = f"{r.get('mean_shift_pct',0):.1f}%"
                        row["KS Stat"]    = r.get("ks_statistic", "")
                    else:
                        row["Train Cats"] = r.get("train_categories", "")
                        row["Test Cats"]  = r.get("test_categories", "")
                        row["Chi2"]       = r.get("chi2", "")
                    dr_display.append(row)

                dr_df = pd.DataFrame(dr_display)
                st.dataframe(dr_df, use_container_width=True, hide_index=True)
    else:
        st.info("Upload a test or production CSV above to compare distributions against training data.")
        st.markdown("""
**Why this matters:**
- If your test data has different distributions than training data, your model will underperform in production
- This is called **covariate shift** — one of the most common causes of ML model failure after deployment
- Features with high drift should be either re-engineered or the model should be retrained with more recent data
        """)

# ── TAB: PREDICTIONS ─────────────────────────────────────────
with tab_predict:
    best_pipeline_pred = model_results.get("best_pipeline")
    label_encoder_pred = model_results.get("label_encoder")
    num_feats          = model_results.get("numeric_features", [])
    cat_feats          = model_results.get("categorical_features", [])
    pred_std           = best_model.get("pred_std", 0.0)

    if best_pipeline_pred is None:
        st.warning("No trained model available for predictions.")
    else:
        st.subheader("🎯 Make Predictions")
        st.caption(f"Using: **{best_model.get('model','?')}** (best model from leaderboard)")

        pred_mode = st.radio("Prediction mode", ["Upload new CSV", "Fill a form"], horizontal=True)

        if pred_mode == "Upload new CSV":
            pred_file = st.file_uploader("Upload CSV without target column", type=["csv"], key="pred_upload")
            if pred_file:
                df_new = pd.read_csv(pred_file)
                st.dataframe(df_new.head(5), use_container_width=True)
                if st.button("Run Predictions", type="primary"):
                    result_df = predict_from_dataframe(
                        df_new, best_pipeline_pred, label_encoder_pred, task_type,
                        num_feats, cat_feats, pred_std,
                    )
                    st.success(f"✅ Predictions complete — {len(result_df):,} rows")
                    st.dataframe(result_df, use_container_width=True)
                    csv_out = result_df.to_csv(index=False).encode("utf-8")
                    st.download_button("⬇️ Download Predictions CSV", csv_out,
                                       "predictions.csv", "text/csv")
        else:
            st.markdown("Fill in the feature values:")
            all_feats     = num_feats + cat_feats
            input_dict: dict = {}
            cols_per_row  = 3
            rows_f        = [all_feats[i:i+cols_per_row] for i in range(0, len(all_feats), cols_per_row)]
            for row_feats in rows_f:
                row_cols = st.columns(len(row_feats))
                for col_widget, feat in zip(row_cols, row_feats):
                    if feat in num_feats:
                        input_dict[feat] = col_widget.number_input(feat, value=0.0, key=f"inp_{feat}")
                    else:
                        unique_vals = df[feat].dropna().unique().tolist() if feat in df.columns else []
                        if unique_vals:
                            input_dict[feat] = col_widget.selectbox(feat, unique_vals, key=f"inp_{feat}")
                        else:
                            input_dict[feat] = col_widget.text_input(feat, key=f"inp_{feat}")

            if st.button("Predict", type="primary"):
                pred_out = predict_single_row(
                    input_dict, best_pipeline_pred, label_encoder_pred, task_type,
                    num_feats, cat_feats, pred_std,
                )
                st.success(f"**Prediction: {pred_out['prediction']}**")
                if "interval" in pred_out:
                    st.info(f"95% Confidence Interval: {pred_out['interval']}")
                if "confidence" in pred_out:
                    st.info(f"Prediction confidence: {pred_out['confidence']*100:.1f}%")
                if pred_out.get("probabilities"):
                    st.markdown("**Class probabilities:**")
                    for cls, prob in pred_out["probabilities"].items():
                        st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

# ── TAB: EXPORT ──────────────────────────────────────────────
with tab_export:
    st.subheader("🧹 Download Cleaned Dataset")
    if cleaned_df is not None and cleaning_report:
        orig  = cleaning_report.get("original_shape", ("?","?"))
        final = cleaning_report.get("final_shape",    ("?","?"))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Original Rows", f"{orig[0]:,}")
        c2.metric("Cleaned Rows",  f"{final[0]:,}", delta=f"-{cleaning_report.get('rows_removed',0)}")
        c3.metric("Original Cols", str(orig[1]))
        c4.metric("Cleaned Cols",  str(final[1]),  delta=f"-{cleaning_report.get('cols_removed',0)}")

        for action in cleaning_report.get("actions", []):
            bullet(f"✅ {action}", "bullet-green")

        csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Cleaned Dataset (CSV)", csv_bytes,
                           "cleaned_dataset.csv", "text/csv", use_container_width=True)

    st.divider()
    st.subheader("📄 Download PDF Diagnostic Report")
    if pdf_bytes:
        st.download_button("⬇️ Download PDF Report", pdf_bytes,
                           "automl_report.pdf", "application/pdf",
                           use_container_width=True, type="primary")
    else:
        st.info("PDF report not available for clustering mode.")

    st.divider()
    st.subheader("💾 Download Trained Model")
    if model_bytes:
        st.download_button(
            "⬇️ Download Trained Model (.joblib)",
            model_bytes,
            "automl_model.joblib",
            "application/octet-stream",
            use_container_width=True,
        )
        st.caption("Load with: `import joblib; bundle = joblib.load('automl_model.joblib')` "
                   "then use `bundle['pipeline'].predict(X_new)`")
    else:
        st.info("Model download not available for clustering mode.")

    st.divider()
    st.subheader("📋 Model Card")
    if model_card_md:
        st.download_button(
            "⬇️ Download Model Card (Markdown)",
            model_card_md.encode("utf-8"),
            "model_card.md",
            "text/markdown",
            use_container_width=True,
        )
        with st.expander("Preview Model Card"):
            st.markdown(model_card_md)

st.divider()
st.caption("AutoML Debugger v3.0 · Stage 2 · Groq LLaMA 3.3 70B · XGBoost · LightGBM · SHAP · scikit-learn")
