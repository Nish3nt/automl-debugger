"""
AutoML Debugger  v3.0  —  Page-Router Architecture
===================================================
Pages: Upload → Preprocess → Train → Results → Predict → Export
Clean sidebar navigation. Minimal industry-grade UI.
"""

import os, time
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

FALLBACK = Path("data/initial_dataset.csv")

# ─────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────
PAGES = ["Upload", "Preprocess", "Train", "Results", "Predict", "Export"]
ICONS = ["📁", "⚙️", "🚀", "📊", "🎯", "📄"]

defaults = {
    "page": 0,
    "df": None,
    "target_column": None,
    "task_type": None,
    "task_reason": None,
    "no_target": False,
    "ts_info": {},
    "profile": {},
    "leakage": {},
    "prep_steps": [],
    "cleaned_df": None,
    "cleaning_report": {},
    "model_results": {},
    "feature_importance": {},
    "leaderboard": [],
    "best_model": {},
    "health": {},
    "llm_analysis": {},
    "pdf_bytes": None,
    "shap_data": {},
    "residual_analysis": {},
    "confusion_analysis": {},
    "feature_suggestions": [],
    "model_card_md": "",
    "model_bytes": None,
    "opt_dup": True,
    "opt_impute": True,
    "opt_out": True,
    "opt_const": True,
    "api_key": "",
    "training_done": False,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go_to(page_idx):
    st.session_state["page"] = page_idx

S = st.session_state   # shorthand

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"]  { background:#0e1117; }
[data-testid="stSidebar"]           { background:#111620; border-right:1px solid #1e2535; }
[data-testid="stSidebar"] *         { font-family:'Segoe UI',sans-serif; }

/* nav buttons */
.nav-btn {
    display:block; width:100%; padding:9px 14px; margin-bottom:4px;
    border-radius:8px; border:none; cursor:pointer; text-align:left;
    font-size:0.88rem; font-weight:500; transition:background .15s;
}
.nav-active  { background:#1e3a5c; color:#7eb6ff; border-left:3px solid #7eb6ff; }
.nav-done    { background:#1a2a1a; color:#5dbc8a; }
.nav-locked  { background:transparent; color:#3a4255; cursor:default; }
.nav-ready   { background:#1a1f2e; color:#8892a4; }
.nav-ready:hover { background:#1e2540; color:#c0c8d8; }

/* KPI cards */
.kpi { background:#141920; border:1px solid #1e2535; border-radius:10px;
       padding:14px 18px; text-align:center; }
.kpi-v { font-size:1.7rem; font-weight:700; color:#7eb6ff; }
.kpi-l { font-size:0.72rem; color:#5a6478; text-transform:uppercase;
          letter-spacing:.8px; margin-top:3px; }

/* cards */
.card  { background:#141920; border:1px solid #1e2535; border-radius:10px;
         padding:16px 20px; margin-bottom:10px; }
.card-green { border-color:#2a5c3a; background:#111a14; }
.card-amber { border-color:#5c4a1a; background:#1a1710; }
.card-red   { border-color:#5c1a1a; background:#1a1010; }

/* top nav bar */
.topnav { display:flex; gap:6px; padding:10px 0 18px 0; }
.tnav-done   { background:#1a2a1a; color:#5dbc8a; border:1px solid #2a5c3a;
               border-radius:20px; padding:5px 14px; font-size:0.82rem; font-weight:600; }
.tnav-active { background:#1e3a5c; color:#7eb6ff; border:1px solid #2a5c8c;
               border-radius:20px; padding:5px 14px; font-size:0.82rem; font-weight:600; }
.tnav-locked { background:#111620; color:#2a3245; border:1px solid #1e2535;
               border-radius:20px; padding:5px 14px; font-size:0.82rem; }

/* model rank cards */
.rank1 { background:#1a1a0a; border:1px solid #c8a84b; border-radius:10px; padding:14px 18px; margin-bottom:8px; }
.rankn { background:#141920; border:1px solid #1e2535; border-radius:10px; padding:14px 18px; margin-bottom:8px; }

/* prep table row */
.prow { display:flex; align-items:center; gap:10px; padding:8px 12px;
        border-bottom:1px solid #1e2535; font-size:0.87rem; }
.pbadge { display:inline-block; padding:2px 9px; border-radius:10px;
          font-size:0.72rem; font-weight:700; white-space:nowrap; }
.pb-imp { background:#0d2a42; color:#5a9fd4; }
.pb-enc { background:#1e0d42; color:#9070d4; }
.pb-drp { background:#421010; color:#d47070; }
.pb-scl { background:#0d2a1a; color:#5abd8a; }
.pb-ts  { background:#2a2a0d; color:#d4d470; }

/* health bar */
.hbar-bg { background:#1a1f2e; border-radius:6px; height:8px; overflow:hidden; margin:4px 0 2px; }

/* fe suggestion */
.fe-card { border-radius:8px; padding:12px 16px; margin-bottom:8px; }

/* progress line */
.pline { background:#141920; border-radius:5px; padding:6px 12px; margin-bottom:4px;
         font-family:monospace; font-size:0.83rem; color:#5a9fd4; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:16px 4px 12px'>"
        "<span style='font-size:1.4rem;font-weight:800;color:#e0e8ff'>🧠 AutoML</span>"
        "<span style='font-size:1.4rem;font-weight:300;color:#5a6478'> Debugger</span>"
        "<div style='font-size:0.72rem;color:#3a4255;margin-top:2px'>v3.0 · Stage 2</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#1e2535;margin:0 0 10px'>", unsafe_allow_html=True)

    # Navigation
    completed = {
        0: S["df"] is not None,
        1: bool(S["prep_steps"]),
        2: S["training_done"],
        3: S["training_done"],
        4: S["training_done"],
        5: S["training_done"],
    }

    for i, (icon, label) in enumerate(zip(ICONS, PAGES)):
        is_active = S["page"] == i
        is_done   = completed.get(i, False) and not is_active
        is_locked = i > 0 and not completed.get(i - 1, False) and not is_active

        if is_active:
            cls = "nav-active"
        elif is_done:
            cls = "nav-done"
        elif is_locked:
            cls = "nav-locked"
        else:
            cls = "nav-ready"

        prefix = "✅ " if is_done else ("🔒 " if is_locked else "")
        if not is_locked:
            if st.button(f"{prefix}{icon} {label}", key=f"nav_{i}",
                         use_container_width=True,
                         disabled=is_locked):
                go_to(i)
                st.rerun()
        else:
            st.markdown(
                f'<div class="nav-btn nav-locked">🔒 {icon} {label}</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<hr style='border-color:#1e2535;margin:10px 0'>", unsafe_allow_html=True)

    # API Key — read from secrets/env silently, allow user to override
    env_key = ""
    if hasattr(st, "secrets"):
        env_key = st.secrets.get("GROQ_API_KEY", "")
    if not env_key:
        env_key = os.environ.get("GROQ_API_KEY", "")

    # Only show input if no key found in environment
    if env_key:
        S["api_key"] = env_key
        st.markdown(
            "<div style='font-size:0.78rem;color:#2a5c3a;background:#111a14;"
            "border:1px solid #2a5c3a;border-radius:6px;padding:7px 10px;margin-bottom:4px'>"
            "🔑 Groq API Key ✅ Connected</div>",
            unsafe_allow_html=True,
        )
    else:
        api_key = st.text_input("🔑 Groq API Key", type="password",
                                 placeholder="gsk_...", label_visibility="visible")
        S["api_key"] = api_key

    st.markdown("<div style='font-size:0.72rem;color:#3a4255;margin-top:-8px;margin-bottom:10px'>Optional — enables LLM analysis</div>", unsafe_allow_html=True)

    # Cleaning options (compact)
    st.markdown("<div style='font-size:0.78rem;color:#5a6478;font-weight:600;margin-bottom:6px'>⚙️ CLEANING</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    S["opt_dup"]    = c1.checkbox("Dupes",    value=S["opt_dup"],    key="cb_dup")
    S["opt_impute"] = c2.checkbox("Impute",   value=S["opt_impute"], key="cb_imp")
    S["opt_out"]    = c1.checkbox("Outliers", value=S["opt_out"],    key="cb_out")
    S["opt_const"]  = c2.checkbox("Constant", value=S["opt_const"],  key="cb_con")

    st.markdown("<hr style='border-color:#1e2535;margin:10px 0'>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.72rem;color:#2a3245;text-align:center'>"
        "Built by <a href='https://github.com/nishantdiwate' style='color:#3a5278'>Nishant Diwate</a>"
        "</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────
# TOP NAV BAR
# ─────────────────────────────────────────────
cur = S["page"]
nav_html = '<div class="topnav">'
for i, (icon, label) in enumerate(zip(ICONS, PAGES)):
    done   = completed.get(i, False) and i != cur
    active = i == cur
    locked = i > 0 and not completed.get(i - 1, False) and not active
    if active:
        cls = "tnav-active"
    elif done:
        cls = "tnav-done"
    else:
        cls = "tnav-locked"
    prefix = "✅ " if done else ""
    nav_html += f'<span class="{cls}">{prefix}{icon} {label}</span>'
nav_html += "</div>"
st.markdown(nav_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def kpi(col, val, label):
    col.markdown(
        f'<div class="kpi"><div class="kpi-v">{val}</div>'
        f'<div class="kpi-l">{label}</div></div>',
        unsafe_allow_html=True,
    )

def card(text, style=""):
    cls = f"card {style}"
    st.markdown(f'<div class="{cls}">{text}</div>', unsafe_allow_html=True)

BADGE = {"imputation":"pb-imp","encoding":"pb-enc","drop":"pb-drp","scaling":"pb-scl","time-series":"pb-ts"}
BLABEL = {"imputation":"IMPUTE","encoding":"ENCODE","drop":"DROP","scaling":"SCALE","time-series":"TS"}


# ══════════════════════════════════════════════
# PAGE 0 — UPLOAD
# ══════════════════════════════════════════════
if cur == 0:
    st.markdown("## 📁 Upload Dataset")

    uploaded = st.file_uploader("CSV file", type=["csv"], label_visibility="collapsed")

    df = None
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if df.shape[0] < 10 or df.shape[1] < 2:
                st.error("Need at least 10 rows and 2 columns.")
                df = None
            elif df.shape[0] > 500_000:
                df = df.sample(500_000, random_state=42)
        except Exception as e:
            st.error(f"Could not read file: {e}")
    elif FALLBACK.exists():
        df = pd.read_csv(FALLBACK)
        st.info(f"Using built-in sample dataset")

    if df is not None:
        S["df"] = df

        # KPIs
        k1,k2,k3,k4 = st.columns(4)
        kpi(k1, f"{df.shape[0]:,}", "Rows")
        kpi(k2, str(df.shape[1]), "Columns")
        kpi(k3, f"{df.isna().sum().sum():,}", "Missing")
        kpi(k4, str(df.duplicated().sum()), "Duplicates")

        st.markdown("<br>", unsafe_allow_html=True)

        # Target + Task
        col_l, col_r = st.columns([1,1])
        with col_l:
            no_target = st.checkbox("No target column (clustering)", value=S["no_target"])
            S["no_target"] = no_target
            if not no_target:
                target = st.selectbox("Target column", df.columns.tolist(),
                                       index=len(df.columns)-1)
                S["target_column"] = target
            else:
                S["target_column"] = None

        with col_r:
            if not no_target and S["target_column"]:
                auto_type, auto_reason = detect_task_type(df[S["target_column"]])
                task = st.radio("Task type", ["regression","classification"],
                                index=0 if auto_type=="regression" else 1,
                                horizontal=True)
                S["task_type"]   = task
                S["task_reason"] = auto_reason
                color = "#5dbc8a" if task == auto_type else "#e0b070"
                st.markdown(
                    f'<div style="background:#141920;border-left:3px solid {color};'
                    f'border-radius:6px;padding:8px 12px;font-size:0.82rem;color:#8892a4;margin-top:6px">'
                    f'{auto_reason}</div>', unsafe_allow_html=True,
                )
            else:
                S["task_type"] = "clustering"

        st.markdown("<br>", unsafe_allow_html=True)

        # Preview
        with st.expander("Preview data"):
            st.dataframe(df.head(8), use_container_width=True)

        st.button("Continue →", type="primary", on_click=lambda: go_to(1))
    else:
        st.markdown(
            '<div class="card" style="text-align:center;color:#3a4255;padding:40px">'
            'Upload a CSV file to begin</div>',
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
# PAGE 1 — PREPROCESS
# ══════════════════════════════════════════════
elif cur == 1:
    st.markdown("## ⚙️ Preprocessing Plan")

    df     = S["df"]
    target = S["target_column"]
    task   = S["task_type"]

    if df is None:
        st.warning("Go back and upload a dataset first.")
        st.stop()

    ts_info = detect_timeseries(df)
    S["ts_info"] = ts_info

    if task != "clustering" and target:
        steps = build_preprocessing_summary(df, target, ts_info.get("datetime_column"))
        S["prep_steps"] = steps

        if ts_info.get("is_timeseries"):
            st.markdown(
                f'<div class="card card-amber">⏱️ Time-series detected — '
                f'<code>{ts_info["datetime_column"]}</code> ({ts_info["frequency_guess"]}). '
                f'Chronological split will be used.</div>',
                unsafe_allow_html=True,
            )

        # Compact table
        st.markdown(
            '<div style="background:#141920;border:1px solid #1e2535;border-radius:10px;overflow:hidden">',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="prow" style="background:#0d1117;font-size:0.75rem;color:#3a4255;font-weight:700">'
            '<span style="width:180px">COLUMN</span>'
            '<span style="width:90px">TYPE</span>'
            '<span>ACTION</span></div>',
            unsafe_allow_html=True,
        )
        for step in steps:
            bc  = BADGE.get(step["category"], "pb-scl")
            bl  = BLABEL.get(step["category"], step["category"].upper())
            st.markdown(
                f'<div class="prow">'
                f'<span style="width:180px;color:#c0c8d8;font-weight:500">{step["column"][:24]}</span>'
                f'<span style="width:90px"><span class="pbadge {bc}">{bl}</span></span>'
                f'<span style="color:#8892a4;font-size:0.85rem">{step["action"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)

    else:
        S["prep_steps"] = ["clustering"]
        st.info("Clustering mode — numeric features will be scaled. No target column.")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,5])
    c1.button("← Back",      on_click=lambda: go_to(0))
    c2.button("Run Pipeline →", type="primary", on_click=lambda: go_to(2))


# ══════════════════════════════════════════════
# PAGE 2 — TRAIN
# ══════════════════════════════════════════════
elif cur == 2:
    st.markdown("## 🚀 Training Pipeline")

    df     = S["df"]
    target = S["target_column"]
    task   = S["task_type"]
    ts     = S["ts_info"]

    if df is None:
        st.warning("Go back and upload a dataset.")
        st.stop()

    if S["training_done"]:
        st.success("✅ Training already complete. Go to Results.")
        c1,c2 = st.columns([1,5])
        c1.button("← Back", on_click=lambda: go_to(1))
        c2.button("View Results →", type="primary", on_click=lambda: go_to(3))
        st.stop()

    log_box   = st.empty()
    log_lines : list[str] = []

    def log(msg):
        log_lines.append(msg)
        html = "".join(f'<div class="pline">{l}</div>' for l in log_lines[-25:])
        log_box.markdown(html, unsafe_allow_html=True)
        time.sleep(0.04)

    log("🚀 Pipeline started...")

    # Profile + leakage
    if task != "clustering" and target:
        log("🔍 Profiling dataset...")
        profile = profile_dataset(df, target)
        S["profile"] = profile
        log(f"✅ {profile['rows']:,} rows · {profile['missing_pct']}% missing · {profile['duplicate_rows']} dupes")

        log("🚨 Checking for data leakage...")
        leakage = detect_leakage(df, target)
        S["leakage"] = leakage
        if leakage["leakage_candidates"]:
            log(f"🚨 Leakage risk: {leakage['leakage_candidates']}")
        else:
            log("✅ No leakage detected.")
    else:
        profile = {}
        leakage = {}
        S["profile"] = {}
        S["leakage"] = {}

    # Cleaning
    log("🧹 Cleaning dataset...")
    if task != "clustering" and target:
        cleaned_df, cleaning_report = clean_dataset(
            df, target,
            remove_duplicates=S["opt_dup"],
            impute_missing=S["opt_impute"],
            cap_outliers=S["opt_out"],
            drop_constant=S["opt_const"],
        )
        for a in cleaning_report.get("actions", []):
            log(f"  ✅ {a}")
    else:
        cleaned_df     = df.copy()
        cleaning_report = {}
    S["cleaned_df"]       = cleaned_df
    S["cleaning_report"]  = cleaning_report

    # Model training
    log("🤖 Starting model training...")
    model_results = run_model_pipeline(
        df=cleaned_df,
        target_column=target,
        task_type=task,
        is_timeseries=ts.get("is_timeseries", False),
        ts_datetime_col=ts.get("datetime_column"),
        progress_callback=log,
    )
    S["model_results"]     = model_results
    S["feature_importance"]= model_results.get("feature_importance", {})
    S["leaderboard"]       = model_results.get("leaderboard", [])
    S["best_model"]        = S["leaderboard"][0] if S["leaderboard"] else {}

    # Health score
    best = S["best_model"]
    if task != "clustering":
        bm = ({"best_r2": best.get("cv_score",0), "r2": best.get("test_r2",0)}
              if task == "regression"
              else {"best_accuracy": best.get("cv_score",0), "accuracy": best.get("test_accuracy",0)})
        health = compute_health_score(profile, bm, task, leakage)
        S["health"] = health
    else:
        S["health"] = {}

    # LLM analysis
    log("🤖 Generating expert analysis...")
    if task != "clustering":
        llm = generate_llm_analysis(
            profile, model_results, task, S["health"],
            leakage, ts, S["feature_importance"],
            S["prep_steps"] if isinstance(S["prep_steps"], list) else [],
            api_key=S["api_key"] or None,
        )
        S["llm_analysis"] = llm
    else:
        S["llm_analysis"] = {}

    # PDF
    if task != "clustering":
        S["pdf_bytes"] = generate_pdf_report(
            profile, model_results, task, S["health"],
            S["llm_analysis"], S["feature_importance"],
            leakage, ts, cleaning_report,
        )

    # Stage 2 computations
    bp = model_results.get("best_pipeline")
    le = model_results.get("label_encoder")
    nf = model_results.get("numeric_features", [])
    cf = model_results.get("categorical_features", [])

    if bp and task != "clustering":
        log("🔮 Computing SHAP values...")
        try:
            X_shap = cleaned_df.drop(columns=[target]).head(200)
            S["shap_data"] = compute_shap_values(bp, X_shap, nf, cf)
        except Exception:
            S["shap_data"] = {"available": False}

        Xt = model_results.get("X_test")
        yt = model_results.get("y_test")
        if Xt is not None and yt is not None:
            yp = bp.predict(Xt)
            if task == "regression":
                S["residual_analysis"]  = compute_residual_analysis(yt, yp, best.get("model",""))
            else:
                S["confusion_analysis"] = compute_confusion_analysis(yt, yp, le, best.get("model",""))

        S["feature_suggestions"] = suggest_feature_engineering(
            cleaned_df, target, S["feature_importance"], profile, task,
        )
        S["model_card_md"] = generate_model_card(
            best.get("model","?"), task, target, profile, S["health"],
            S["leaderboard"], S["feature_importance"],
            leakage, ts, cleaning_report,
        )
        try:
            S["model_bytes"] = save_model_to_bytes(bp, le, nf, cf, task, best.get("model",""), target)
        except Exception:
            S["model_bytes"] = None

    S["training_done"] = True
    log("✅ All done!")

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("View Results →", type="primary", on_click=lambda: go_to(3))


# ══════════════════════════════════════════════
# PAGE 3 — RESULTS
# ══════════════════════════════════════════════
elif cur == 3:
    if not S["training_done"]:
        st.warning("Run the pipeline first.")
        st.button("← Go to Train", on_click=lambda: go_to(2))
        st.stop()

    task    = S["task_type"]
    health  = S["health"]
    profile = S["profile"]
    leakage = S["leakage"]
    ts      = S["ts_info"]
    lb      = S["leaderboard"]
    best    = S["best_model"]
    fi      = S["feature_importance"]
    llm     = S["llm_analysis"]
    mr      = S["model_results"]

    # ── CLUSTERING ────────────────────────────
    if task == "clustering":
        st.markdown("## 📊 Clustering Results")
        best_k = mr.get("best_k","?")
        sizes  = mr.get("cluster_sizes",{})
        k1,k2,k3 = st.columns(3)
        kpi(k1, str(best_k), "Optimal Clusters")
        kpi(k2, str(max(sizes.values()) if sizes else "?"), "Largest Cluster")
        kpi(k3, f"{mr.get('inertia',0):.0f}", "Inertia")

        c1,c2 = st.columns(2)
        with c1:
            size_df = pd.DataFrame(list(sizes.items()), columns=["Cluster","Size"])
            fig = px.bar(size_df, x="Cluster", y="Size", template="plotly_dark",
                         color_discrete_sequence=["#7eb6ff"])
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#141920",
                              margin=dict(t=20,b=20))
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            elbow = mr.get("elbow_data",{})
            if elbow:
                fig2 = go.Figure(go.Scatter(x=elbow["k_values"], y=elbow["inertias"],
                    mode="lines+markers", line=dict(color="#7eb6ff",width=2)))
                fig2.update_layout(template="plotly_dark", paper_bgcolor="#0e1117",
                    plot_bgcolor="#141920", margin=dict(t=20,b=20),
                    xaxis_title="k", yaxis_title="Inertia")
                st.plotly_chart(fig2, use_container_width=True)

        prof = mr.get("profiles")
        if prof is not None:
            st.markdown("**Cluster Profiles**")
            st.dataframe(prof, use_container_width=True, hide_index=True)
        st.stop()

    # ── SUPERVISED RESULTS ────────────────────
    total   = health.get("total", 0)
    verdict = health.get("verdict","")
    bar_col = "#5dbc8a" if total>=80 else ("#e0b070" if total>=60 else "#e07070")
    vclass  = "card-green" if total>=80 else ("card-amber" if total>=60 else "card-red")

    # Verdict banner
    st.markdown(
        f'<div class="card {vclass}" style="display:flex;align-items:center;gap:20px">'
        f'<div style="font-size:2.2rem;font-weight:800;color:{bar_col}">{total}</div>'
        f'<div><div style="font-size:0.7rem;color:#5a6478;text-transform:uppercase;letter-spacing:1px">Health Score / 100</div>'
        f'<div style="font-size:1rem;font-weight:600;color:#c0c8d8;margin-top:2px">{verdict}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Quick flags
    flags = []
    if leakage.get("leakage_candidates"):
        flags.append(f'<span style="background:#421010;color:#e07070;padding:3px 10px;border-radius:12px;font-size:0.78rem;font-weight:600;margin-right:5px">🚨 {len(leakage["leakage_candidates"])} Leakage Risk</span>')
    if ts.get("is_timeseries"):
        flags.append(f'<span style="background:#0d2a42;color:#5a9fd4;padding:3px 10px;border-radius:12px;font-size:0.78rem;font-weight:600;margin-right:5px">⏱️ Time-Series</span>')
    if profile.get("missing_pct",0) > 10:
        flags.append(f'<span style="background:#2a2000;color:#d4a040;padding:3px 10px;border-radius:12px;font-size:0.78rem;font-weight:600;margin-right:5px">⚠️ {profile["missing_pct"]}% Missing</span>')
    if flags:
        st.markdown("<div style='margin:8px 0'>" + "".join(flags) + "</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPI row
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpi(k1, f"{profile.get('rows',0):,}", "Rows")
    kpi(k2, str(profile.get('columns',0)), "Columns")
    kpi(k3, f"{profile.get('missing_pct',0)}%", "Missing")
    kpi(k4, str(profile.get('duplicate_rows',0)), "Duplicates")
    kpi(k5, task.capitalize(), "Task")
    if task == "regression":
        kpi(k6, str(best.get("cv_score","—")), "Best CV R²")
    else:
        kpi(k6, str(best.get("cv_score","—")), "Best CV Acc")

    st.markdown("<br>", unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────
    tabs = st.tabs([
        "📊 Overview", "🏆 Models", "🌲 Features",
        "🔮 SHAP", "📉 Evaluation", "🔍 Data Quality",
        "🚨 Leakage", "⚙️ Feature Eng.", "📡 Drift",
    ])

    # ── Overview ─────────────────────────────
    with tabs[0]:
        col_a, col_b = st.columns([1,1])

        with col_a:
            st.markdown("**Health Breakdown**")
            for dim, dv in health.get("dimensions",{}).items():
                pct = int(dv["score"]/dv["max"]*100)
                c = "#5dbc8a" if pct>=80 else ("#e0b070" if pct>=50 else "#e07070")
                st.markdown(
                    f'<div style="margin-bottom:10px">'
                    f'<div style="display:flex;justify-content:space-between;font-size:0.82rem;color:#8892a4;margin-bottom:3px">'
                    f'<span>{dim}</span><span style="color:{c}">{dv["score"]}/{dv["max"]}</span></div>'
                    f'<div class="hbar-bg"><div style="width:{pct}%;background:{c};height:100%;border-radius:6px"></div></div>'
                    f'<div style="font-size:0.75rem;color:#3a4255;margin-top:2px">{dv["reason"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with col_b:
            st.markdown("**Action Plan**")
            for b in llm.get("action_plan", []):
                st.markdown(
                    f'<div class="card card-amber" style="padding:9px 13px;font-size:0.87rem;margin-bottom:6px">{b}</div>',
                    unsafe_allow_html=True,
                )

        st.markdown("**Expert Analysis**")
        for b in llm.get("overall_analysis", []):
            st.markdown(
                f'<div class="card" style="padding:9px 13px;font-size:0.87rem;margin-bottom:6px">{b}</div>',
                unsafe_allow_html=True,
            )

    # ── Models ───────────────────────────────
    with tabs[1]:
        baseline = mr.get("baseline",{})
        if baseline:
            bl_val = baseline.get("r2","?") if task=="regression" else baseline.get("accuracy","?")
            bl_key = "Baseline R²" if task=="regression" else "Baseline Accuracy"
            st.markdown(
                f'<div class="card card-amber" style="font-size:0.85rem;padding:10px 14px">'
                f'📏 <b>{bl_key}: {bl_val}</b> — {baseline.get("description","")} — your model must beat this.</div>',
                unsafe_allow_html=True,
            )

        for i, m in enumerate(lb):
            cls = "rank1" if i==0 else "rankn"
            badge = "🥇" if i==0 else ("🥈" if i==1 else "🥉")
            gap   = abs((m.get("train_score") or 0) - (m.get("cv_score") or 0))
            gap_c = "#e07070" if gap > 0.1 else "#5dbc8a"
            gap_t = "⚠️ Overfit risk" if gap > 0.1 else "✅ Good generalization"

            if task == "regression":
                metrics = (f"CV R²&nbsp;<b>{m.get('cv_score','?')}</b> &nbsp;·&nbsp; "
                           f"Train R²&nbsp;<b>{m.get('train_score','?')}</b> &nbsp;·&nbsp; "
                           f"Test R²&nbsp;<b>{m.get('test_r2','?')}</b> &nbsp;·&nbsp; "
                           f"MAE&nbsp;<b>{m.get('mae','?')}</b> &nbsp;·&nbsp; "
                           f"RMSE&nbsp;<b>{m.get('rmse','?')}</b>")
            else:
                metrics = (f"CV Acc&nbsp;<b>{m.get('cv_score','?')}</b> &nbsp;·&nbsp; "
                           f"Train&nbsp;<b>{m.get('train_score','?')}</b> &nbsp;·&nbsp; "
                           f"Test&nbsp;<b>{m.get('test_accuracy','?')}</b> &nbsp;·&nbsp; "
                           f"F1&nbsp;<b>{m.get('f1_score','?')}</b> &nbsp;·&nbsp; "
                           f"AUC&nbsp;<b>{m.get('roc_auc','?')}</b>")

            st.markdown(
                f'<div class="{cls}"><div style="font-size:0.95rem;font-weight:700;color:#e0e0e0;margin-bottom:5px">'
                f'{badge} {m.get("model","?")}</div>'
                f'<div style="font-size:0.83rem;color:#5a6478">{metrics}</div>'
                f'<div style="font-size:0.78rem;margin-top:5px;color:{gap_c}">Train/CV gap={gap:.4f} — {gap_t}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Model commentary
        for b in llm.get("model_commentary",[]):
            st.markdown(f'<div class="card" style="padding:9px 13px;font-size:0.86rem;margin-bottom:5px">{b}</div>',
                        unsafe_allow_html=True)

        # Tier 1 chart
        t1 = [r for r in mr.get("tier1_results",[]) if r.get("cv_score",-999) > -999]
        if t1:
            t1_df = pd.DataFrame(t1)[["model","cv_score"]].rename(columns={"model":"Model","cv_score":"CV Score"})
            fig = px.bar(t1_df.sort_values("CV Score"), x="CV Score", y="Model",
                         orientation="h", template="plotly_dark",
                         color="CV Score", color_continuous_scale="Blues")
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#141920",
                              coloraxis_showscale=False, margin=dict(t=10,b=10), height=200)
            st.plotly_chart(fig, use_container_width=True)

    # ── Features ─────────────────────────────
    with tabs[2]:
        if fi:
            fi_df = pd.DataFrame({"Feature":list(fi.keys()),"Importance":list(fi.values())}).sort_values("Importance",ascending=True)
            fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                         color="Importance", color_continuous_scale="Blues", template="plotly_dark")
            fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#141920",
                              coloraxis_showscale=False, margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

            for b in llm.get("feature_commentary",[]):
                st.markdown(f'<div class="card" style="padding:9px 13px;font-size:0.86rem;margin-bottom:5px">{b}</div>',
                            unsafe_allow_html=True)

            if profile.get("top_correlations"):
                c_df = pd.DataFrame({"Feature":list(profile["top_correlations"].keys()),
                                     "Correlation":list(profile["top_correlations"].values())}).sort_values("Correlation",key=abs,ascending=True)
                fig2 = px.bar(c_df, x="Correlation", y="Feature", orientation="h",
                              color="Correlation", color_continuous_scale="RdBu",
                              color_continuous_midpoint=0, template="plotly_dark")
                fig2.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#141920",
                                   coloraxis_showscale=False, margin=dict(t=10))
                st.plotly_chart(fig2, use_container_width=True)

    # ── SHAP ─────────────────────────────────
    with tabs[3]:
        shap_d = S.get("shap_data",{})
        if not shap_d.get("available"):
            st.warning(shap_d.get("error","SHAP not available."))
        else:
            gi = shap_d.get("global_importance",{})
            if gi:
                gi_df = pd.DataFrame({"Feature":list(gi.keys()),"Mean |SHAP|":list(gi.values())}).sort_values("Mean |SHAP|",ascending=True)
                fig = px.bar(gi_df, x="Mean |SHAP|", y="Feature", orientation="h",
                             color="Mean |SHAP|", color_continuous_scale="Blues", template="plotly_dark")
                fig.update_layout(paper_bgcolor="#0e1117", plot_bgcolor="#141920",
                                  coloraxis_showscale=False, margin=dict(t=10))
                st.plotly_chart(fig, use_container_width=True)

            for s in shap_d.get("sample_explanations",[]):
                with st.expander(f"Row {s['row']+1} — prediction drivers"):
                    for feat in s["top_features"]:
                        c = "#5dbc8a" if feat["shap"]>0 else "#e07070"
                        d = "▲ pushed UP" if feat["shap"]>0 else "▼ pushed DOWN"
                        st.markdown(
                            f'<div class="card" style="padding:7px 12px;border-left:3px solid {c};margin-bottom:4px;font-size:0.85rem">'
                            f'<b>{feat["feature"]}</b> — SHAP {feat["shap"]:+.4f} — <span style="color:{c}">{d}</span></div>',
                            unsafe_allow_html=True,
                        )

    # ── Evaluation ───────────────────────────
    with tabs[4]:
        if task == "regression":
            ra = S.get("residual_analysis",{})
            if ra:
                stats = ra.get("stats",{})
                r1,r2,r3,r4 = st.columns(4)
                kpi(r1, str(stats.get("mean_residual","—")), "Mean Residual")
                kpi(r2, str(stats.get("mae","—")), "MAE")
                kpi(r3, f"{stats.get('within_10pct','?')}%", "Within 10%")
                kpi(r4, f"{stats.get('within_20pct','?')}%", "Within 20%")

                for interp in ra.get("interpretations",[]):
                    c = "card-red" if "⚠️" in interp or "🔴" in interp else ("card-green" if "✅" in interp else "")
                    st.markdown(f'<div class="card {c}" style="padding:9px 13px;font-size:0.86rem;margin-bottom:5px">{interp}</div>',
                                unsafe_allow_html=True)

                pd_  = ra.get("plot_data",{})
                if pd_:
                    col_a, col_b = st.columns(2)
                    with col_a:
                        fig_ap = px.scatter(x=pd_["y_true"], y=pd_["y_pred"],
                                            labels={"x":"Actual","y":"Predicted"},
                                            template="plotly_dark", color_discrete_sequence=["#7eb6ff"])
                        mn = min(min(pd_["y_true"]),min(pd_["y_pred"]))
                        mx = max(max(pd_["y_true"]),max(pd_["y_pred"]))
                        fig_ap.add_shape(type="line",x0=mn,x1=mx,y0=mn,y1=mx,
                                         line=dict(color="#e07070",dash="dash"))
                        fig_ap.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#141920",
                                             margin=dict(t=10),title="Actual vs Predicted")
                        st.plotly_chart(fig_ap, use_container_width=True)
                    with col_b:
                        fig_rp = px.scatter(x=pd_["y_pred"], y=pd_["residuals"],
                                            labels={"x":"Predicted","y":"Residual"},
                                            template="plotly_dark", color_discrete_sequence=["#e0b070"])
                        fig_rp.add_hline(y=0,line_color="#e07070",line_dash="dash")
                        fig_rp.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#141920",
                                             margin=dict(t=10),title="Residuals vs Predicted")
                        st.plotly_chart(fig_rp, use_container_width=True)

                    fig_rh = px.histogram(x=pd_["residuals"],nbins=40,template="plotly_dark",
                                          color_discrete_sequence=["#7eb6ff"],
                                          labels={"x":"Residual"})
                    fig_rh.add_vline(x=0,line_color="#e07070",line_dash="dash")
                    fig_rh.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#141920",
                                         margin=dict(t=10),title="Residual Distribution")
                    st.plotly_chart(fig_rh, use_container_width=True)
        else:
            ca = S.get("confusion_analysis",{})
            if ca:
                for interp in ca.get("interpretations",[]):
                    c = "card-red" if "🔴" in interp or "⚠️" in interp else ("card-green" if "✅" in interp else "")
                    st.markdown(f'<div class="card {c}" style="padding:9px 13px;font-size:0.86rem;margin-bottom:5px">{interp}</div>',
                                unsafe_allow_html=True)
                cm_data   = ca.get("confusion_matrix",[])
                classes_c = ca.get("classes",[])
                if cm_data and classes_c:
                    col_a, col_b = st.columns([1,1])
                    with col_a:
                        cm_arr = np.array(cm_data)
                        fig_cm = px.imshow(cm_arr, x=classes_c, y=classes_c,
                                           color_continuous_scale="Blues", template="plotly_dark",
                                           labels={"x":"Predicted","y":"Actual"}, text_auto=True)
                        fig_cm.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#141920",margin=dict(t=10))
                        st.plotly_chart(fig_cm, use_container_width=True)
                    with col_b:
                        pc_df = pd.DataFrame(ca.get("per_class",[]))
                        st.dataframe(pc_df, use_container_width=True, hide_index=True)

    # ── Data Quality ─────────────────────────
    with tabs[5]:
        df_raw = S["df"]
        col_a, col_b = st.columns(2)
        with col_a:
            ms = df_raw.isna().sum()
            ms = ms[ms>0]
            if not ms.empty:
                mdf = ms.reset_index()
                mdf.columns = ["Column","Missing"]
                fig = px.bar(mdf, x="Missing", y="Column", orientation="h",
                             color="Missing", color_continuous_scale="Reds", template="plotly_dark")
                fig.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#141920",
                                  coloraxis_showscale=False,margin=dict(t=10),title="Missing Values")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.success("No missing values.")
        with col_b:
            target = S["target_column"]
            if target and target in df_raw.columns:
                if task == "classification":
                    vc = df_raw[target].value_counts().reset_index()
                    vc.columns = ["Class","Count"]
                    fig = px.pie(vc, names="Class", values="Count", template="plotly_dark",
                                 color_discrete_sequence=px.colors.sequential.Blues_r)
                else:
                    fig = px.histogram(df_raw, x=target, template="plotly_dark",
                                       color_discrete_sequence=["#7eb6ff"], nbins=40)
                fig.update_layout(paper_bgcolor="#0e1117",plot_bgcolor="#141920",margin=dict(t=10),title="Target Distribution")
                st.plotly_chart(fig, use_container_width=True)

        oc = profile.get("outlier_counts",{})
        if oc:
            oc_df = pd.DataFrame({"Feature":list(oc.keys()),"Outliers":list(oc.values())}).sort_values("Outliers",ascending=False)
            st.dataframe(oc_df, use_container_width=True, hide_index=True)

    # ── Leakage ──────────────────────────────
    with tabs[6]:
        if leakage.get("leakage_candidates"):
            st.error(f"{len(leakage['leakage_candidates'])} feature(s) with >0.95 correlation with target: {leakage['leakage_candidates']}")
            for w in leakage.get("warnings",[]):
                st.markdown(f'<div class="card card-red" style="padding:9px 13px;font-size:0.86rem;margin-bottom:5px">{w}</div>',
                            unsafe_allow_html=True)
        else:
            st.success("No leakage features detected.")

        hcf = leakage.get("high_correlation_features",{})
        if hcf:
            cs = sorted(hcf.items(), key=lambda x: abs(x[1]), reverse=True)
            cdf = pd.DataFrame(cs, columns=["Feature","Abs Correlation"])
            cdf["Risk"] = cdf["Abs Correlation"].apply(
                lambda x: "🚨 HIGH" if x>0.95 else ("⚠️ Watch" if x>0.85 else "✅ OK"))
            st.dataframe(cdf, use_container_width=True, hide_index=True)

        if ts.get("is_timeseries"):
            st.markdown("<hr style='border-color:#1e2535'>", unsafe_allow_html=True)
            col_a, col_b = st.columns(2)
            col_a.metric("DateTime Column",  ts.get("datetime_column","—"))
            col_b.metric("Frequency",        ts.get("frequency_guess","—"))

    # ── Feature Engineering ──────────────────
    with tabs[7]:
        sugs = S.get("feature_suggestions",[])
        if not sugs:
            st.success("No critical feature engineering needed.")
        else:
            pc = {"HIGH":"#e07070","MEDIUM":"#e0b070","LOW":"#7eb6ff"}
            pb = {"HIGH":"#2e1a1a","MEDIUM":"#2e2a1a","LOW":"#1a2a3a"}
            for sug in sugs:
                pri = sug.get("priority","LOW")
                st.markdown(
                    f'<div class="fe-card" style="background:{pb.get(pri,"#141920")};'
                    f'border-left:3px solid {pc.get(pri,"#7eb6ff")};margin-bottom:8px">'
                    f'<span style="color:{pc.get(pri)};font-size:0.72rem;font-weight:700">{pri}</span> '
                    f'<span style="color:#3a4255;font-size:0.72rem">{sug["type"]}</span><br>'
                    f'<b style="color:#c0c8d8;font-size:0.9rem">{sug["suggestion"]}</b><br>'
                    f'<span style="color:#5a6478;font-size:0.82rem">{sug["reason"]}</span><br>'
                    f'<code style="background:#0e1117;padding:5px 8px;border-radius:4px;'
                    f'font-size:0.78rem;color:#7eb6ff;display:block;margin-top:6px">{sug["code_snippet"]}</code>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Drift ────────────────────────────────
    with tabs[8]:
        drift_file = st.file_uploader("Upload test/production CSV", type=["csv"], key="drift_up")
        if drift_file:
            df_drift = pd.read_csv(drift_file)
            st.caption(f"{df_drift.shape[0]:,} rows × {df_drift.shape[1]} cols")
            if st.button("Run Drift Analysis", type="primary"):
                with st.spinner("Analysing..."):
                    from src.explainer import detect_drift
                    dr = detect_drift(S["cleaned_df"], df_drift, S["target_column"])
                vd  = dr.get("verdict","")
                vc  = "#e07070" if "HIGH" in vd else ("#e0b070" if "MODERATE" in vd else "#5dbc8a")
                vcl = "card-red" if "HIGH" in vd else ("card-amber" if "MODERATE" in vd else "card-green")
                st.markdown(f'<div class="card {vcl}" style="padding:10px 14px;font-size:0.9rem">{vd}</div>',
                            unsafe_allow_html=True)
                dr_list = dr.get("drift_results",[])
                if dr_list:
                    rows = []
                    for r in dr_list:
                        rows.append({
                            "Feature":   r["feature"],
                            "Drift":     "🔴 YES" if r["drift_detected"] else "🟢 NO",
                            "Severity":  r.get("severity",""),
                            "p-value":   r.get("p_value",""),
                        })
                    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("Upload a test or production CSV to compare against training distributions.")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,5])
    c1.button("← Back", on_click=lambda: go_to(2))
    c2.button("Make Predictions →", type="primary", on_click=lambda: go_to(4))


# ══════════════════════════════════════════════
# PAGE 4 — PREDICT
# ══════════════════════════════════════════════
elif cur == 4:
    if not S["training_done"]:
        st.warning("Run the pipeline first.")
        st.button("← Go to Train", on_click=lambda: go_to(2))
        st.stop()

    st.markdown("## 🎯 Predictions")

    mr   = S["model_results"]
    bp   = mr.get("best_pipeline")
    le   = mr.get("label_encoder")
    nf   = mr.get("numeric_features",[])
    cf   = mr.get("categorical_features",[])
    std  = S["best_model"].get("pred_std", 0.0)
    task = S["task_type"]

    if bp is None:
        st.warning("No trained model available.")
        st.stop()

    st.caption(f"Model: **{S['best_model'].get('model','?')}** · CV score: **{S['best_model'].get('cv_score','?')}**")

    mode = st.radio("", ["📂 Upload CSV", "📝 Fill form"], horizontal=True, label_visibility="collapsed")

    if mode == "📂 Upload CSV":
        pf = st.file_uploader("CSV without target column", type=["csv"], key="pred_up")
        if pf:
            df_new = pd.read_csv(pf)
            st.dataframe(df_new.head(5), use_container_width=True)
            if st.button("Run Predictions", type="primary"):
                out = predict_from_dataframe(df_new, bp, le, task, nf, cf, std)
                st.dataframe(out, use_container_width=True)
                st.download_button("⬇️ Download CSV", out.to_csv(index=False).encode(),
                                   "predictions.csv", "text/csv")
    else:
        all_f = nf + cf
        indict: dict = {}
        cols_pr = 3
        for row_f in [all_f[i:i+cols_pr] for i in range(0,len(all_f),cols_pr)]:
            rcs = st.columns(len(row_f))
            for rc, feat in zip(rcs, row_f):
                if feat in nf:
                    indict[feat] = rc.number_input(feat, value=0.0, key=f"p_{feat}")
                else:
                    uv = S["df"][feat].dropna().unique().tolist() if feat in S["df"].columns else []
                    indict[feat] = rc.selectbox(feat, uv, key=f"p_{feat}") if uv else rc.text_input(feat, key=f"p_{feat}")

        if st.button("Predict", type="primary"):
            out = predict_single_row(indict, bp, le, task, nf, cf, std)
            pred_val = out["prediction"]
            st.markdown(
                f'<div class="card card-green" style="font-size:1.5rem;font-weight:700;text-align:center;padding:20px">'
                f'Prediction: {pred_val}</div>',
                unsafe_allow_html=True,
            )
            if "interval" in out:
                st.info(f"95% CI: {out['interval']}")
            if "confidence" in out:
                st.info(f"Confidence: {out['confidence']*100:.1f}%")
            if out.get("probabilities"):
                for cls, prob in out["probabilities"].items():
                    st.progress(float(prob), text=f"{cls}: {prob*100:.1f}%")

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,5])
    c1.button("← Results", on_click=lambda: go_to(3))
    c2.button("Export →", type="primary", on_click=lambda: go_to(5))


# ══════════════════════════════════════════════
# PAGE 5 — EXPORT
# ══════════════════════════════════════════════
elif cur == 5:
    if not S["training_done"]:
        st.warning("Run the pipeline first.")
        st.button("← Go to Train", on_click=lambda: go_to(2))
        st.stop()

    st.markdown("## 📄 Export")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**Cleaned Dataset**")
        cdf = S.get("cleaned_df")
        cr  = S.get("cleaning_report",{})
        if cdf is not None:
            orig  = cr.get("original_shape",("?","?"))
            final = cr.get("final_shape",("?","?"))
            st.caption(f"{orig[0]}×{orig[1]} → {final[0]}×{final[1]}")
            st.download_button("⬇️ cleaned_dataset.csv",
                               cdf.to_csv(index=False).encode(),
                               "cleaned_dataset.csv", "text/csv",
                               use_container_width=True)

        st.markdown("<br>**Trained Model**", unsafe_allow_html=True)
        mb = S.get("model_bytes")
        if mb:
            st.download_button("⬇️ automl_model.joblib", mb,
                               "automl_model.joblib", "application/octet-stream",
                               use_container_width=True)
            st.caption("`bundle = joblib.load('automl_model.joblib')` → `bundle['pipeline'].predict(X)`")
        else:
            st.info("Not available for clustering.")

    with c2:
        st.markdown("**PDF Report**")
        pb = S.get("pdf_bytes")
        if pb:
            st.download_button("⬇️ automl_report.pdf", pb,
                               "automl_report.pdf", "application/pdf",
                               use_container_width=True, type="primary")
        else:
            st.info("Not available for clustering.")

        st.markdown("<br>**Model Card**", unsafe_allow_html=True)
        mc = S.get("model_card_md","")
        if mc:
            st.download_button("⬇️ model_card.md",
                               mc.encode("utf-8"),
                               "model_card.md", "text/markdown",
                               use_container_width=True)
            with st.expander("Preview"):
                st.markdown(mc)
        else:
            st.info("Not available for clustering.")

    st.markdown("<br>", unsafe_allow_html=True)
    st.button("← Predictions", on_click=lambda: go_to(4))

st.markdown(
    "<div style='text-align:center;color:#1e2535;font-size:0.72rem;margin-top:30px'>"
    "AutoML Debugger v3.0 · Groq LLaMA 3.3 70B · XGBoost · LightGBM · SHAP</div>",
    unsafe_allow_html=True,
)
