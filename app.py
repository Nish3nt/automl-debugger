"""
AutoML Debugger — Streamlit Application
========================================
Industry-grade ML dataset diagnostics with LLM-powered expert analysis.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── must be first Streamlit call ──────────────────────────────────────────────
st.set_page_config(
    page_title="AutoML Debugger",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── import engine ─────────────────────────────────────────────────────────────
from src.debugger_engine import run_debugger_pipeline

# ─────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── General ── */
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    [data-testid="stSidebar"]          { background-color: #161b27; border-right: 1px solid #2a2f45; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }

    /* ── Metric cards ── */
    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #1e2540 100%);
        border: 1px solid #2e3555;
        border-radius: 12px;
        padding: 18px 22px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7eb6ff; }
    .metric-label { font-size: 0.8rem; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

    /* ── Analysis bullets ── */
    .analysis-bullet {
        background: #161b27;
        border-left: 3px solid #7eb6ff;
        border-radius: 6px;
        padding: 12px 16px;
        margin-bottom: 10px;
        font-size: 0.95rem;
        line-height: 1.5;
    }

    /* ── Health score bar ── */
    .health-container { margin-top: 12px; }
    .health-bar-bg {
        background: #1a1f35;
        border-radius: 8px;
        height: 16px;
        overflow: hidden;
        border: 1px solid #2e3555;
    }

    /* ── Info pills ── */
    .pill {
        display: inline-block;
        padding: 3px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 3px;
    }
    .pill-blue  { background:#1a3a5c; color:#7eb6ff; border:1px solid #2a5080; }
    .pill-green { background:#1a3a2c; color:#5dbc8a; border:1px solid #2a6040; }
    .pill-red   { background:#3a1a1a; color:#e07070; border:1px solid #804040; }
    .pill-amber { background:#3a2a1a; color:#e0b070; border:1px solid #806040; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.divider()

    # API key — stored in st.secrets for Streamlit Cloud, else manual input
    api_key = st.secrets.get("ANTHROPIC_API_KEY", "") if hasattr(st, "secrets") else ""
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "🔑 Anthropic API Key (optional)",
            type="password",
            help="Enables real LLM-powered expert analysis via Claude. Leave blank for rule-based analysis.",
        )

    st.divider()
    st.markdown("### 📌 About")
    st.markdown("""
AutoML Debugger evaluates your dataset **before** you train any model.

**Checks performed:**
- Missing values & duplicates
- Outlier detection (IQR)
- Class imbalance
- Feature correlations
- Baseline + RF model metrics
- Cross-validated performance
- LLM expert commentary
    """)
    st.divider()
    st.markdown("Built by [Nishant Diwate](https://github.com/nishantdiwate)")

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.markdown("<h1 style='font-size:3rem;margin:0'>🧠</h1>", unsafe_allow_html=True)
with col_title:
    st.markdown("<h1 style='margin:0;padding-top:10px'>AutoML Debugger</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#8892a4;margin:0'>LLM-Assisted Dataset Diagnostics for ML Engineers</p>", unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────────────
# Dataset upload
# ─────────────────────────────────────────────
FALLBACK_DATASET = Path("data/initial_dataset.csv")

col_up, col_info = st.columns([2, 1])

with col_up:
    st.subheader("📂 Upload Dataset")
    uploaded_file = st.file_uploader(
        "Upload a CSV file — or run with the built-in sample dataset",
        type=["csv"],
        label_visibility="collapsed",
    )

with col_info:
    st.markdown("""
    **Supported formats:** CSV  
    **Min size:** 10 rows, 2 columns  
    **Tip:** The last column is auto-selected as the target  
    """)

# Load dataset
df, source = None, None
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    source = "uploaded"
elif FALLBACK_DATASET.exists():
    df = pd.read_csv(FALLBACK_DATASET)
    source = "fallback"

if source == "uploaded":
    st.success(f"✅ Uploaded dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")
elif source == "fallback":
    st.info(f"ℹ️ Using built-in sample dataset — {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# Dataset preview + target selection
# ─────────────────────────────────────────────
target_column = None
if df is not None:
    with st.expander("🔍 Preview Dataset", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")

    target_column = st.selectbox(
        "🎯 Select Target Column",
        options=df.columns.tolist(),
        index=len(df.columns) - 1,
        help="This is the column your model will predict.",
    )

st.divider()

# ─────────────────────────────────────────────
# Run button
# ─────────────────────────────────────────────
run_clicked = st.button("🚀 Run AutoML Diagnostics", type="primary", use_container_width=True)

if run_clicked:
    if df is None:
        st.warning("No dataset available. Please upload a CSV file.")
        st.stop()

    with st.spinner("🔬 Running full ML diagnostics pipeline…"):
        output = run_debugger_pipeline(df, target_column, api_key=api_key or None)

    metrics          = output.get("metrics", {})
    profile          = output.get("profile", {})
    feature_imp      = output.get("feature_importance", {})
    llm_analysis     = output.get("llm_analysis", [])
    task_type        = output.get("task_type", "unknown")
    health_score     = output.get("health_score", 0)
    diagnosis        = output.get("diagnosis", "")

    if not metrics:
        st.error(f"❌ {diagnosis}")
        for line in llm_analysis:
            st.warning(line)
        st.stop()

    # ─────────────────────────────────────────
    # Top KPI row
    # ─────────────────────────────────────────
    st.markdown("---")
    kpi_cols = st.columns(5)

    def kpi(col, value, label):
        col.markdown(
            f'<div class="metric-card"><div class="metric-value">{value}</div>'
            f'<div class="metric-label">{label}</div></div>',
            unsafe_allow_html=True,
        )

    kpi(kpi_cols[0], f"{metrics.get('rows', 0):,}", "Rows")
    kpi(kpi_cols[1], str(metrics.get('columns', 0)), "Columns")
    kpi(kpi_cols[2], f"{metrics.get('missing_values', 0):,}", "Missing Values")
    kpi(kpi_cols[3], str(profile.get('duplicate_rows', 0)), "Duplicates")
    kpi(kpi_cols[4], task_type.capitalize(), "Task Type")

    st.markdown("<br>", unsafe_allow_html=True)

    # ─────────────────────────────────────────
    # Health score
    # ─────────────────────────────────────────
    if health_score >= 80:
        bar_color, verdict = "#5dbc8a", "🟢 Production-Ready"
    elif health_score >= 60:
        bar_color, verdict = "#e0b070", "🟡 Needs Minor Work"
    else:
        bar_color, verdict = "#e07070", "🔴 Significant Issues"

    st.subheader(f"⭐ Dataset Health Score — {verdict}")
    st.markdown(
        f"""
        <div class="health-container">
          <div class="health-bar-bg">
            <div style="width:{health_score}%;background:{bar_color};height:100%;border-radius:8px;
                        transition:width 0.6s ease;"></div>
          </div>
          <p style="color:{bar_color};font-size:1.4rem;font-weight:700;margin-top:8px">{health_score} / 100</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    # ─────────────────────────────────────────
    # Tabbed detail view
    # ─────────────────────────────────────────
    tab_metrics, tab_analysis, tab_features, tab_data_quality = st.tabs([
        "📊 Model Metrics",
        "🧠 Expert Analysis",
        "🌲 Feature Importance",
        "🔍 Data Quality",
    ])

    # ── Tab 1: Model Metrics ──────────────────
    with tab_metrics:
        st.subheader("Model Performance Metrics")
        st.caption(f"Task type auto-detected as: **{task_type}**  |  {diagnosis}")

        if task_type == "regression":
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("R² (Baseline)", metrics.get("r2_baseline", "—"))
            m2.metric("R² (Random Forest)", metrics.get("r2_rf", "—"))
            m3.metric("MAE", metrics.get("mae", "—"))
            m4.metric("RMSE", metrics.get("rmse", "—"))
            cv_mean = metrics.get("cv_r2_mean", "—")
            cv_std  = metrics.get("cv_r2_std",  "—")
            m5.metric("CV R² (5-fold)", f"{cv_mean} ± {cv_std}")
        else:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy (Baseline)", metrics.get("accuracy_baseline", "—"))
            m2.metric("Accuracy (RF)", metrics.get("accuracy_rf", "—"))
            m3.metric("F1 Score", metrics.get("f1_score", "—"))
            m4.metric("ROC-AUC", metrics.get("roc_auc", "—"))
            cv_mean = metrics.get("cv_accuracy_mean", "—")
            cv_std  = metrics.get("cv_accuracy_std",  "—")
            m5.metric("CV Accuracy (5-fold)", f"{cv_mean} ± {cv_std}")

        st.divider()

        # Radar chart of normalised metrics
        if task_type == "regression":
            raw = {
                "R² Score":     max(0, metrics.get("r2_rf", 0)),
                "Low MAE":      max(0, 1 - min(1, metrics.get("mae", 0) / max(df[target_column].std(), 1))),
                "Data Density": min(1, metrics.get("rows", 0) / 10000),
                "Completeness": 1 - profile.get("missing_pct", 0) / 100,
                "No Duplicates":1 - min(1, profile.get("duplicate_rows", 0) / max(metrics.get("rows", 1), 1)),
            }
        else:
            raw = {
                "Accuracy":     metrics.get("accuracy_rf", 0),
                "F1 Score":     metrics.get("f1_score", 0),
                "ROC-AUC":      metrics.get("roc_auc", metrics.get("accuracy_rf", 0)),
                "Completeness": 1 - profile.get("missing_pct", 0) / 100,
                "No Duplicates":1 - min(1, profile.get("duplicate_rows", 0) / max(metrics.get("rows", 1), 1)),
            }

        cats   = list(raw.keys())
        values = list(raw.values())

        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]],
            theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(126,182,255,0.2)",
            line=dict(color="#7eb6ff", width=2),
            name="Dataset Quality",
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1a1f35",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2e3555", color="#8892a4"),
                angularaxis=dict(gridcolor="#2e3555", color="#e0e0e0"),
            ),
            showlegend=False,
            paper_bgcolor="#0e1117",
            margin=dict(l=60, r=60, t=40, b=40),
            height=360,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Tab 2: Expert Analysis ────────────────
    with tab_analysis:
        if api_key:
            st.caption("🤖 Analysis powered by **Claude (Anthropic)** LLM")
        else:
            st.caption("⚙️ Rule-based analysis — add an Anthropic API key in the sidebar for LLM-powered commentary")

        for point in llm_analysis:
            st.markdown(
                f'<div class="analysis-bullet">{point}</div>',
                unsafe_allow_html=True,
            )

    # ── Tab 3: Feature Importance ─────────────
    with tab_features:
        if feature_imp:
            fi_df = pd.DataFrame({
                "Feature":    list(feature_imp.keys()),
                "Importance": list(feature_imp.values()),
            }).sort_values("Importance", ascending=True)

            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature",
                orientation="h",
                title="Top Features by Random Forest Importance",
                color="Importance",
                color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig_fi.update_layout(
                paper_bgcolor="#0e1117",
                plot_bgcolor="#1a1f35",
                coloraxis_showscale=False,
                yaxis=dict(gridcolor="#2e3555"),
                xaxis=dict(gridcolor="#2e3555"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            # Top correlations with target
            if profile.get("top_correlations"):
                st.subheader("📈 Top Feature Correlations with Target")
                corr_df = pd.DataFrame({
                    "Feature":     list(profile["top_correlations"].keys()),
                    "Correlation": list(profile["top_correlations"].values()),
                }).sort_values("Correlation", key=abs, ascending=True)

                fig_corr = px.bar(
                    corr_df, x="Correlation", y="Feature",
                    orientation="h",
                    color="Correlation",
                    color_continuous_scale="RdBu",
                    color_continuous_midpoint=0,
                    template="plotly_dark",
                )
                fig_corr.update_layout(
                    paper_bgcolor="#0e1117",
                    plot_bgcolor="#1a1f35",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Feature importance could not be computed for this dataset.")

    # ── Tab 4: Data Quality ───────────────────
    with tab_data_quality:
        q1, q2 = st.columns(2)

        with q1:
            st.subheader("🧩 Missing Values per Column")
            missing_series = df.isna().sum()
            missing_series = missing_series[missing_series > 0]
            if not missing_series.empty:
                miss_df = missing_series.reset_index()
                miss_df.columns = ["Column", "Missing Count"]
                fig_miss = px.bar(
                    miss_df, x="Missing Count", y="Column",
                    orientation="h", template="plotly_dark",
                    color="Missing Count", color_continuous_scale="Reds",
                )
                fig_miss.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("✅ No missing values found!")

        with q2:
            st.subheader("📊 Target Distribution")
            if task_type == "classification":
                vc = df[target_column].value_counts().reset_index()
                vc.columns = ["Class", "Count"]
                fig_dist = px.pie(
                    vc, names="Class", values="Count",
                    template="plotly_dark",
                    color_discrete_sequence=px.colors.sequential.Blues_r,
                )
            else:
                fig_dist = px.histogram(
                    df, x=target_column,
                    template="plotly_dark",
                    color_discrete_sequence=["#7eb6ff"],
                    nbins=40,
                )
            fig_dist.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35")
            st.plotly_chart(fig_dist, use_container_width=True)

        # Outlier summary
        if profile.get("outlier_counts"):
            st.subheader("⚠️ Outlier Summary (IQR method)")
            out_df = pd.DataFrame({
                "Feature":       list(profile["outlier_counts"].keys()),
                "Outlier Count": list(profile["outlier_counts"].values()),
            }).sort_values("Outlier Count", ascending=False)
            st.dataframe(out_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No significant outliers detected!")

        # Quality flags
        st.subheader("🏷️ Data Quality Flags")
        flags = []
        if profile.get("constant_features"):
            flags.append(f'<span class="pill pill-red">⚠️ {len(profile["constant_features"])} constant feature(s)</span>')
        if profile.get("high_cardinality_cols"):
            flags.append(f'<span class="pill pill-amber">⚠️ {len(profile["high_cardinality_cols"])} high-cardinality column(s)</span>')
        if profile.get("duplicate_rows", 0) > 0:
            flags.append(f'<span class="pill pill-amber">⚠️ {profile["duplicate_rows"]} duplicate rows</span>')
        if profile.get("missing_pct", 0) > 10:
            flags.append(f'<span class="pill pill-red">⚠️ {profile["missing_pct"]}% missing values</span>')
        if profile.get("imbalance_ratio", 0) and profile["imbalance_ratio"] > 3:
            flags.append(f'<span class="pill pill-amber">⚠️ Class imbalance ratio {profile["imbalance_ratio"]}:1</span>')

        if flags:
            st.markdown(" ".join(flags), unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill-green">✅ No critical data quality flags</span>', unsafe_allow_html=True)

    st.divider()
    st.caption("AutoML Debugger · Built with Streamlit, scikit-learn & Anthropic Claude")