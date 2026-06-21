"""
AutoML Debugger — Streamlit Application  (v2.0 — Industry-Ready)
=================================================================
Advancements:
  1. Groq API  (replaces Anthropic)
  2. Dataset cleaning + export
  3. Data leakage detection panel
  4. Time-series detection warnings
  5. PDF diagnostic report download
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

# ── must be first Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="AutoML Debugger",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

from src.debugger_engine import run_debugger_pipeline, clean_dataset

# ─────────────────────────────────────────────
# Custom CSS  (dark theme — unchanged)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background-color: #0e1117; }
    [data-testid="stSidebar"]          { background-color: #161b27; border-right: 1px solid #2a2f45; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }

    .metric-card {
        background: linear-gradient(135deg, #1a1f35 0%, #1e2540 100%);
        border: 1px solid #2e3555; border-radius: 12px;
        padding: 18px 22px; text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #7eb6ff; }
    .metric-label { font-size: 0.8rem; color: #8892a4; text-transform: uppercase; letter-spacing: 1px; margin-top: 4px; }

    .analysis-bullet {
        background: #161b27; border-left: 3px solid #7eb6ff;
        border-radius: 6px; padding: 12px 16px;
        margin-bottom: 10px; font-size: 0.95rem; line-height: 1.5;
    }
    .leakage-bullet {
        background: #2a1a1a; border-left: 3px solid #e07070;
        border-radius: 6px; padding: 12px 16px;
        margin-bottom: 10px; font-size: 0.92rem; line-height: 1.5;
    }
    .ts-bullet {
        background: #1a2a2a; border-left: 3px solid #5dbc8a;
        border-radius: 6px; padding: 12px 16px;
        margin-bottom: 10px; font-size: 0.92rem; line-height: 1.5;
    }
    .cleaning-action {
        background: #1a2a1a; border-left: 3px solid #5dbc8a;
        border-radius: 6px; padding: 10px 14px;
        margin-bottom: 8px; font-size: 0.9rem;
    }

    .health-container { margin-top: 12px; }
    .health-bar-bg {
        background: #1a1f35; border-radius: 8px;
        height: 16px; overflow: hidden; border: 1px solid #2e3555;
    }

    .pill {
        display: inline-block; padding: 3px 12px;
        border-radius: 20px; font-size: 0.8rem;
        font-weight: 600; margin: 3px;
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

    # ── ADVANCEMENT 1: Groq API key ──────────
    api_key = ""
    if hasattr(st, "secrets"):
        api_key = st.secrets.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        api_key = st.text_input(
            "🔑 Groq API Key (optional)",
            type="password",
            help="Enables LLM-powered expert analysis via Groq LLaMA 3.3 70B. Leave blank for rule-based analysis.",
        )

    st.divider()

    # ── ADVANCEMENT 2: Cleaning options ──────
    st.markdown("### 🧹 Dataset Cleaning Options")
    run_cleaning      = st.toggle("Auto-clean dataset after diagnosis", value=True)
    opt_duplicates    = st.checkbox("Remove duplicate rows",       value=True)
    opt_impute        = st.checkbox("Impute missing values",       value=True)
    opt_outliers      = st.checkbox("Cap outliers (IQR method)",   value=True)
    opt_constant      = st.checkbox("Drop constant columns",       value=True)

    st.divider()
    st.markdown("### 📌 About")
    st.markdown("""
AutoML Debugger v2.0 evaluates your dataset **before** you train any model.

**Checks performed:**
- Missing values & duplicates
- Outlier detection (IQR)
- Class imbalance
- ⏱️ Time-series detection
- 🚨 Data leakage detection
- Baseline + RF model metrics
- Cross-validated performance
- 🤖 Groq LLM expert commentary
- 🧹 One-click dataset cleaning
- 📄 PDF report export
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
    st.markdown("<p style='color:#8892a4;margin:0'>LLM-Assisted Dataset Diagnostics for ML Engineers — v2.0</p>", unsafe_allow_html=True)

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

df, source = None, None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        # Basic validation
        if df.shape[0] < 10 or df.shape[1] < 2:
            st.error("❌ File is too small. Need at least 10 rows and 2 columns.")
            df = None
        elif df.shape[0] > 500_000:
            st.warning("⚠️ Large file (>500k rows). Sampling 500k rows for performance.")
            df = df.sample(500_000, random_state=42)
        source = "uploaded"
    except Exception as e:
        st.error(f"❌ Could not read file: {e}")
elif FALLBACK_DATASET.exists():
    df = pd.read_csv(FALLBACK_DATASET)
    source = "fallback"

if df is not None:
    if source == "uploaded":
        st.success(f"✅ Uploaded dataset loaded — {df.shape[0]:,} rows × {df.shape[1]} columns")
    else:
        st.info(f"ℹ️ Using built-in sample dataset — {df.shape[0]:,} rows × {df.shape[1]} columns")

# ─────────────────────────────────────────────
# Dataset preview + target selection
# ─────────────────────────────────────────────
target_column = None
if df is not None:
    with st.expander("🔍 Preview Dataset", expanded=False):
        st.dataframe(df.head(10), use_container_width=True)
        st.caption(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns | Dtypes: {dict(df.dtypes.value_counts())}")

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

    cleaning_options = {
        "remove_duplicates": opt_duplicates,
        "impute_missing":    opt_impute,
        "cap_outliers":      opt_outliers,
        "drop_constant":     opt_constant,
    }

    with st.spinner("🔬 Running full ML diagnostics pipeline…"):
        output = run_debugger_pipeline(
            df,
            target_column,
            api_key=api_key or None,
            run_cleaning=run_cleaning,
            cleaning_options=cleaning_options,
        )

    metrics          = output.get("metrics", {})
    profile          = output.get("profile", {})
    feature_imp      = output.get("feature_importance", {})
    llm_analysis     = output.get("llm_analysis", [])
    task_type        = output.get("task_type", "unknown")
    health_score     = output.get("health_score", 0)
    diagnosis        = output.get("diagnosis", "")
    leakage          = output.get("leakage", {})
    ts_info          = output.get("ts_info", {})
    cleaned_df       = output.get("cleaned_df")
    cleaning_report  = output.get("cleaning_report", {})
    pdf_bytes        = output.get("pdf_bytes")

    if not metrics:
        st.error(f"❌ {diagnosis}")
        for line in llm_analysis:
            st.warning(line)
        st.stop()

    # ─────────────────────────────────────────
    # ADVANCEMENT 4 — Time-series banner
    # ─────────────────────────────────────────
    if ts_info.get("is_timeseries"):
        st.markdown(
            f'<div class="ts-bullet">⏱️ <b>Time-Series Dataset Detected</b> — '
            f'Column: <code>{ts_info["datetime_column"]}</code> | '
            f'Frequency: <b>{ts_info["frequency_guess"]}</b>. '
            f'Chronological train/test split applied to prevent future data leakage.</div>',
            unsafe_allow_html=True,
        )

    # ─────────────────────────────────────────
    # ADVANCEMENT 3 — Leakage banner
    # ─────────────────────────────────────────
    if leakage.get("leakage_candidates"):
        st.markdown(
            f'<div class="leakage-bullet">🚨 <b>Data Leakage Risk Detected!</b> '
            f'{len(leakage["leakage_candidates"])} feature(s) have correlation &gt;0.95 with target: '
            f'<code>{", ".join(leakage["leakage_candidates"])}</code>. '
            f'Verify these are not derived from the target before deployment.</div>',
            unsafe_allow_html=True,
        )

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

    kpi(kpi_cols[0], f"{metrics.get('rows', 0):,}",         "Rows")
    kpi(kpi_cols[1], str(metrics.get('columns', 0)),         "Columns")
    kpi(kpi_cols[2], f"{metrics.get('missing_values', 0):,}","Missing Values")
    kpi(kpi_cols[3], str(profile.get('duplicate_rows', 0)),  "Duplicates")
    kpi(kpi_cols[4], task_type.capitalize(),                  "Task Type")

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
    # Tabs
    # ─────────────────────────────────────────
    tab_metrics, tab_analysis, tab_leakage, tab_features, tab_quality, tab_clean = st.tabs([
        "📊 Model Metrics",
        "🧠 Expert Analysis",
        "🚨 Leakage & Time-Series",
        "🌲 Feature Importance",
        "🔍 Data Quality",
        "🧹 Cleaned Dataset",
    ])

    # ── Tab 1: Model Metrics ──────────────────
    with tab_metrics:
        st.subheader("Model Performance Metrics")
        st.caption(f"Task type auto-detected as: **{task_type}**  |  {diagnosis}")

        if ts_info.get("is_timeseries"):
            st.info("⏱️ TimeSeriesSplit (5-fold) used for cross-validation — no data leakage from the future.")

        if task_type == "regression":
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("R² (Baseline)",      metrics.get("r2_baseline", "—"))
            m2.metric("R² (Random Forest)", metrics.get("r2_rf", "—"))
            m3.metric("MAE",                metrics.get("mae", "—"))
            m4.metric("RMSE",               metrics.get("rmse", "—"))
            cv_mean = metrics.get("cv_r2_mean", "—")
            cv_std  = metrics.get("cv_r2_std",  "—")
            m5.metric("CV R² (5-fold)", f"{cv_mean} ± {cv_std}")
        else:
            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Accuracy (Baseline)", metrics.get("accuracy_baseline", "—"))
            m2.metric("Accuracy (RF)",       metrics.get("accuracy_rf", "—"))
            m3.metric("F1 Score",            metrics.get("f1_score", "—"))
            m4.metric("ROC-AUC",             metrics.get("roc_auc", "—"))
            cv_mean = metrics.get("cv_accuracy_mean", "—")
            cv_std  = metrics.get("cv_accuracy_std",  "—")
            m5.metric("CV Accuracy (5-fold)", f"{cv_mean} ± {cv_std}")

        st.divider()

        # Radar chart
        if task_type == "regression":
            raw = {
                "R² Score":      max(0, metrics.get("r2_rf", 0)),
                "Low MAE":       max(0, 1 - min(1, metrics.get("mae", 0) / max(float(df[target_column].std()), 1))),
                "Data Density":  min(1, metrics.get("rows", 0) / 10000),
                "Completeness":  1 - profile.get("missing_pct", 0) / 100,
                "No Duplicates": 1 - min(1, profile.get("duplicate_rows", 0) / max(metrics.get("rows", 1), 1)),
            }
        else:
            raw = {
                "Accuracy":      metrics.get("accuracy_rf", 0),
                "F1 Score":      metrics.get("f1_score", 0),
                "ROC-AUC":       metrics.get("roc_auc", metrics.get("accuracy_rf", 0)),
                "Completeness":  1 - profile.get("missing_pct", 0) / 100,
                "No Duplicates": 1 - min(1, profile.get("duplicate_rows", 0) / max(metrics.get("rows", 1), 1)),
            }

        cats   = list(raw.keys())
        values = list(raw.values())
        fig_radar = go.Figure(go.Scatterpolar(
            r=values + [values[0]], theta=cats + [cats[0]],
            fill="toself",
            fillcolor="rgba(126,182,255,0.2)",
            line=dict(color="#7eb6ff", width=2),
        ))
        fig_radar.update_layout(
            polar=dict(
                bgcolor="#1a1f35",
                radialaxis=dict(visible=True, range=[0, 1], gridcolor="#2e3555", color="#8892a4"),
                angularaxis=dict(gridcolor="#2e3555", color="#e0e0e0"),
            ),
            showlegend=False, paper_bgcolor="#0e1117",
            margin=dict(l=60, r=60, t=40, b=40), height=360,
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    # ── Tab 2: Expert Analysis ────────────────
    with tab_analysis:
        if api_key:
            st.caption("🤖 Analysis powered by **Groq LLaMA 3.3 70B**")
        else:
            st.caption("⚙️ Rule-based analysis — add a Groq API key in the sidebar for LLM-powered commentary")

        for point in llm_analysis:
            st.markdown(f'<div class="analysis-bullet">{point}</div>', unsafe_allow_html=True)

    # ── Tab 3: Leakage & Time-Series (ADVANCEMENT 3+4) ──
    with tab_leakage:
        # Time-series section
        st.subheader("⏱️ Time-Series Detection")
        if ts_info.get("is_timeseries"):
            col_a, col_b = st.columns(2)
            col_a.metric("DateTime Column",   ts_info.get("datetime_column", "—"))
            col_b.metric("Estimated Frequency", ts_info.get("frequency_guess", "—"))
            for w in ts_info.get("warnings", []):
                st.markdown(f'<div class="ts-bullet">{w}</div>', unsafe_allow_html=True)
        else:
            st.success("✅ No time-series structure detected — standard random split is safe.")

        st.divider()

        # Leakage section
        st.subheader("🚨 Data Leakage Detection")
        if leakage.get("leakage_candidates"):
            st.error(
                f"**{len(leakage['leakage_candidates'])} potential leakage feature(s) found** — "
                f"correlation > 0.95 with target column."
            )
            for w in leakage.get("warnings", []):
                st.markdown(f'<div class="leakage-bullet">{w}</div>', unsafe_allow_html=True)
        else:
            st.success("✅ No data leakage features detected (no feature has correlation > 0.95 with target).")

        # Correlation heatmap table
        if leakage.get("high_correlation_features"):
            st.subheader("📊 All Feature Correlations with Target")
            corr_data = sorted(
                leakage["high_correlation_features"].items(),
                key=lambda x: abs(x[1]), reverse=True,
            )
            corr_df = pd.DataFrame(corr_data, columns=["Feature", "Absolute Correlation"])
            corr_df["Risk Level"] = corr_df["Absolute Correlation"].apply(
                lambda x: "🚨 HIGH RISK" if x > 0.95 else ("⚠️ Watch" if x > 0.85 else "✅ OK")
            )
            st.dataframe(corr_df, use_container_width=True, hide_index=True)

    # ── Tab 4: Feature Importance ─────────────
    with tab_features:
        if feature_imp:
            fi_df = pd.DataFrame({
                "Feature":    list(feature_imp.keys()),
                "Importance": list(feature_imp.values()),
            }).sort_values("Importance", ascending=True)

            fig_fi = px.bar(
                fi_df, x="Importance", y="Feature", orientation="h",
                title="Top Features by Random Forest Importance",
                color="Importance", color_continuous_scale="Blues",
                template="plotly_dark",
            )
            fig_fi.update_layout(
                paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
                coloraxis_showscale=False,
                yaxis=dict(gridcolor="#2e3555"),
                xaxis=dict(gridcolor="#2e3555"),
            )
            st.plotly_chart(fig_fi, use_container_width=True)

            if profile.get("top_correlations"):
                st.subheader("📈 Top Feature Correlations with Target")
                corr_df = pd.DataFrame({
                    "Feature":     list(profile["top_correlations"].keys()),
                    "Correlation": list(profile["top_correlations"].values()),
                }).sort_values("Correlation", key=abs, ascending=True)

                fig_corr = px.bar(
                    corr_df, x="Correlation", y="Feature", orientation="h",
                    color="Correlation", color_continuous_scale="RdBu",
                    color_continuous_midpoint=0, template="plotly_dark",
                )
                fig_corr.update_layout(
                    paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35",
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_corr, use_container_width=True)
        else:
            st.info("Feature importance could not be computed for this dataset.")

    # ── Tab 5: Data Quality ───────────────────
    with tab_quality:
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
                fig_miss.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35", coloraxis_showscale=False)
                st.plotly_chart(fig_miss, use_container_width=True)
            else:
                st.success("✅ No missing values found!")

        with q2:
            st.subheader("📊 Target Distribution")
            if task_type == "classification":
                vc = df[target_column].value_counts().reset_index()
                vc.columns = ["Class", "Count"]
                fig_dist = px.pie(vc, names="Class", values="Count", template="plotly_dark",
                                  color_discrete_sequence=px.colors.sequential.Blues_r)
            else:
                fig_dist = px.histogram(df, x=target_column, template="plotly_dark",
                                        color_discrete_sequence=["#7eb6ff"], nbins=40)
            fig_dist.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#1a1f35")
            st.plotly_chart(fig_dist, use_container_width=True)

        if profile.get("outlier_counts"):
            st.subheader("⚠️ Outlier Summary (IQR method)")
            out_df = pd.DataFrame({
                "Feature":       list(profile["outlier_counts"].keys()),
                "Outlier Count": list(profile["outlier_counts"].values()),
            }).sort_values("Outlier Count", ascending=False)
            st.dataframe(out_df, use_container_width=True, hide_index=True)
        else:
            st.success("✅ No significant outliers detected!")

        # Column type warnings
        if profile.get("type_warnings"):
            st.subheader("🏷️ Column Type Warnings")
            for w in profile["type_warnings"]:
                st.warning(w)

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
        if profile.get("imbalance_ratio") and profile["imbalance_ratio"] > 3:
            flags.append(f'<span class="pill pill-amber">⚠️ Class imbalance {profile["imbalance_ratio"]}:1</span>')
        if leakage.get("leakage_candidates"):
            flags.append(f'<span class="pill pill-red">🚨 {len(leakage["leakage_candidates"])} leakage risk(s)</span>')
        if ts_info.get("is_timeseries"):
            flags.append('<span class="pill pill-blue">⏱️ Time-series detected</span>')

        if flags:
            st.markdown(" ".join(flags), unsafe_allow_html=True)
        else:
            st.markdown('<span class="pill pill-green">✅ No critical data quality flags</span>', unsafe_allow_html=True)

    # ── Tab 6: Cleaned Dataset (ADVANCEMENT 2) ──
    with tab_clean:
        st.subheader("🧹 Cleaned Dataset")

        if not run_cleaning:
            st.info("Dataset cleaning is disabled. Enable it in the sidebar to see the cleaned version.")
        elif cleaned_df is not None and cleaning_report:
            # Before / After comparison
            orig = cleaning_report.get("original_shape", ("?", "?"))
            final = cleaning_report.get("final_shape", ("?", "?"))

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Original Rows",  f"{orig[0]:,}")
            c2.metric("Cleaned Rows",   f"{final[0]:,}", delta=f"{cleaning_report.get('rows_removed', 0) * -1} removed")
            c3.metric("Original Cols",  str(orig[1]))
            c4.metric("Cleaned Cols",   str(final[1]), delta=f"{cleaning_report.get('cols_removed', 0) * -1} removed")

            st.markdown("#### Actions Performed")
            if cleaning_report.get("actions"):
                for action in cleaning_report["actions"]:
                    st.markdown(f'<div class="cleaning-action">✅ {action}</div>', unsafe_allow_html=True)
            else:
                st.success("No cleaning actions needed — dataset was already clean!")

            remaining_missing = cleaning_report.get("missing_after", 0)
            if remaining_missing == 0:
                st.success("✅ Zero missing values remaining after cleaning.")
            else:
                st.warning(f"⚠️ {remaining_missing} missing values remain (non-imputable columns).")

            st.markdown("#### Preview Cleaned Dataset")
            st.dataframe(cleaned_df.head(10), use_container_width=True)
            st.caption(f"Cleaned shape: {cleaned_df.shape[0]:,} rows × {cleaned_df.shape[1]} columns")

            # Download cleaned CSV
            csv_bytes = cleaned_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="⬇️ Download Cleaned Dataset (CSV)",
                data=csv_bytes,
                file_name="cleaned_dataset.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.info("Run diagnostics to generate the cleaned dataset.")

    # ─────────────────────────────────────────
    # ADVANCEMENT 5 — PDF Download (bottom of page)
    # ─────────────────────────────────────────
    st.divider()
    st.subheader("📄 Export Diagnostic Report")

    dl_col, info_col = st.columns([1, 2])
    with dl_col:
        if pdf_bytes:
            st.download_button(
                label="⬇️ Download PDF Report",
                data=pdf_bytes,
                file_name="automl_diagnostic_report.pdf",
                mime="application/pdf",
                use_container_width=True,
                type="primary",
            )
    with info_col:
        st.markdown(
            "The PDF report includes the health score, model metrics, expert analysis, "
            "feature importance, leakage risks, time-series warnings, and cleaning summary."
        )

    st.divider()
    st.caption("AutoML Debugger v2.0 · Powered by Groq LLaMA 3.3 70B · Built with Streamlit & scikit-learn")
