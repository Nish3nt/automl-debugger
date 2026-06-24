"""
AutoML Debugger v4.0 — Milestone 1
====================================
5 Pages: Upload → Audit → Deep Analysis → Scorecard → Report
Clean dark UI. No training. No fake accuracy. Pure dataset intelligence.
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
    detect_task_type, detect_timeseries,
    profile_dataset, detect_leakage,
    compute_health_score,
    analyze_distributions,
    detect_redundancy,
    analyze_missing_patterns,
    infer_column_types,
    check_sample_size,
    build_scorecard,
    apply_automated_fixes,
)
from src.report_generator import generate_groq_report, generate_pdf

FALLBACK = Path("data/initial_dataset.csv")

# ─────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────
PAGES = ["Upload", "Audit", "Deep Analysis", "Scorecard", "Report"]
ICONS = ["📁", "🔍", "🧬", "📊", "📄"]

defaults = {
    "page": 0,
    "df": None,
    "target_column": None,
    "task_type": None,
    "task_reason": None,
    "ts_info": {},
    "profile": {},
    "leakage": {},
    "distributions": {},
    "redundancy": {},
    "missing_pattern": {},
    "type_inference": {},
    "sample_check": {},
    "health": {},
    "scorecard": {},
    "fixed_df": None,
    "fix_actions": [],
    "fix_opts": {},
    "report_sections": {},
    "pdf_bytes": None,
    "analysis_done": False,
    "report_done": False,
    "api_key": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

S = st.session_state

def go_to(i): S["page"] = i

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background:#0a0e1a; }
[data-testid="stSidebar"]          { background:#0d1120; border-right:1px solid #1a2035; }

.kpi { background:#111827; border:1px solid #1e2d45; border-radius:10px;
       padding:14px 18px; text-align:center; margin-bottom:4px; }
.kpi-v { font-size:1.8rem; font-weight:800; color:#3b82f6; line-height:1.1; }
.kpi-l { font-size:0.7rem; color:#475569; text-transform:uppercase;
         letter-spacing:.9px; margin-top:4px; }

.card  { background:#111827; border:1px solid #1e2d45; border-radius:10px;
         padding:14px 18px; margin-bottom:8px; }
.card-green { border-color:#166534; background:#052e16; }
.card-amber { border-color:#92400e; background:#1c1400; }
.card-red   { border-color:#991b1b; background:#1c0a0a; }
.card-blue  { border-color:#1e40af; background:#0c1a3a; }

.grade-a { color:#22c55e; }  .grade-b { color:#86efac; }
.grade-c { color:#f59e0b; }  .grade-d { color:#f97316; }
.grade-f { color:#ef4444; }

.nav-active { background:#1e3a5c !important; color:#60a5fa !important;
              border-left:3px solid #3b82f6 !important; border-radius:6px; }
.nav-done   { background:#052e16 !important; color:#4ade80 !important; border-radius:6px; }
.nav-lock   { color:#1e2d45 !important; cursor:default !important; }

.section-title { font-size:1.3rem; font-weight:700; color:#e2e8f0;
                 margin-bottom:4px; padding-bottom:6px;
                 border-bottom:1px solid #1e2d45; }
.dim-bar-bg { background:#1e2d45; border-radius:4px; height:7px; overflow:hidden; }

.badge { display:inline-block; padding:2px 9px; border-radius:10px;
         font-size:0.72rem; font-weight:700; }
.badge-ok     { background:#052e16; color:#4ade80; }
.badge-warn   { background:#1c1400; color:#fbbf24; }
.badge-danger { background:#1c0a0a; color:#f87171; }
.badge-info   { background:#0c1a3a; color:#60a5fa; }

.pline { background:#111827; border-radius:5px; padding:6px 12px;
         margin-bottom:4px; font-family:monospace; font-size:0.82rem; color:#60a5fa; }

.rep-section { background:#111827; border:1px solid #1e2d45; border-radius:10px;
               padding:18px 22px; margin-bottom:12px; }
.rep-title   { font-size:1rem; font-weight:700; color:#3b82f6; margin-bottom:8px; }
.rep-body    { font-size:0.9rem; color:#94a3b8; line-height:1.7; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def kpi(col, val, label):
    col.markdown(f'<div class="kpi"><div class="kpi-v">{val}</div><div class="kpi-l">{label}</div></div>',
                 unsafe_allow_html=True)

def card(text, style=""):
    st.markdown(f'<div class="card {style}">{text}</div>', unsafe_allow_html=True)

def section_title(text):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)

def badge(text, style="info"):
    return f'<span class="badge badge-{style}">{text}</span>'

def grade_color(g):
    return {"A":"#22c55e","B":"#86efac","C":"#f59e0b","D":"#f97316","F":"#ef4444"}.get(g,"#94a3b8")

def health_bar(score, max_score):
    pct = int(score / max(max_score,1) * 100)
    col = "#22c55e" if pct>=80 else ("#f59e0b" if pct>=50 else "#ef4444")
    return (f'<div class="dim-bar-bg"><div style="width:{pct}%;background:{col};'
            f'height:100%;border-radius:4px"></div></div>')

completed = {
    0: S["df"] is not None,
    1: S["analysis_done"],
    2: S["analysis_done"],
    3: S["analysis_done"],
    4: S["report_done"],
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        "<div style='padding:14px 4px 10px'>"
        "<div style='font-size:1.3rem;font-weight:800;color:#e2e8f0'>🧠 AutoML <span style=\"color:#3b82f6\">Debugger</span></div>"
        "<div style='font-size:0.7rem;color:#334155;margin-top:2px'>ML Dataset Audit Tool · v4.0</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<hr style='border-color:#1e2d45;margin:0 0 8px'>", unsafe_allow_html=True)

    for i, (icon, label) in enumerate(zip(ICONS, PAGES)):
        is_active = S["page"] == i
        is_done   = completed.get(i, False) and not is_active
        is_locked = i > 0 and not completed.get(i-1, False) and not is_active

        prefix = "✅ " if is_done else ""
        if is_locked:
            st.markdown(f'<div style="padding:8px 12px;color:#1e2d45;font-size:0.85rem">🔒 {icon} {label}</div>',
                        unsafe_allow_html=True)
        else:
            cls = "nav-active" if is_active else ("nav-done" if is_done else "")
            if st.button(f"{prefix}{icon} {label}", key=f"nav_{i}", use_container_width=True):
                go_to(i); st.rerun()

    st.markdown("<hr style='border-color:#1e2d45;margin:10px 0'>", unsafe_allow_html=True)

    # API key
    env_key = ""
    if hasattr(st, "secrets"): env_key = st.secrets.get("GROQ_API_KEY","")
    if not env_key: env_key = os.environ.get("GROQ_API_KEY","")
    if env_key:
        S["api_key"] = env_key
        st.markdown('<div style="background:#052e16;border:1px solid #166534;border-radius:6px;'
                    'padding:7px 10px;font-size:0.78rem;color:#4ade80">🔑 Groq API ✅ Connected</div>',
                    unsafe_allow_html=True)
    else:
        key_in = st.text_input("🔑 Groq API Key", type="password", placeholder="gsk_...")
        if key_in: S["api_key"] = key_in

    # Export buttons (only after analysis)
    if S["analysis_done"]:
        st.markdown("<hr style='border-color:#1e2d45;margin:10px 0'>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:0.72rem;color:#334155;font-weight:700;margin-bottom:6px">⬇️ EXPORT</div>',
                    unsafe_allow_html=True)
        if S.get("fixed_df") is not None:
            csv_b = S["fixed_df"].to_csv(index=False).encode()
            st.download_button("📥 Cleaned Dataset", csv_b, "cleaned_dataset.csv",
                               "text/csv", use_container_width=True)
        if S.get("pdf_bytes"):
            st.download_button("📥 PDF Report", S["pdf_bytes"], "automl_report.pdf",
                               "application/pdf", use_container_width=True)

    st.markdown("<hr style='border-color:#1e2d45;margin:10px 0'>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:0.7rem;color:#1e2d45;text-align:center">'
                'Built by <a href="https://github.com/nishantdiwate" style="color:#1e3a5c">Nishant Diwate</a></div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# TOP BREADCRUMB
# ─────────────────────────────────────────────
cur = S["page"]
nav_html = '<div style="display:flex;gap:5px;margin-bottom:18px;flex-wrap:wrap">'
for i, (icon, label) in enumerate(zip(ICONS, PAGES)):
    done   = completed.get(i, False) and i != cur
    active = i == cur
    locked = i > 0 and not completed.get(i-1, False) and not active
    if active:
        bg, fg, bdr = "#1e3a5c", "#60a5fa", "#3b82f6"
    elif done:
        bg, fg, bdr = "#052e16", "#4ade80", "#166534"
    else:
        bg, fg, bdr = "#0d1120", "#1e2d45", "#1e2d45"
    pref = "✅ " if done else ""
    nav_html += (f'<span style="background:{bg};color:{fg};border:1px solid {bdr};'
                 f'border-radius:16px;padding:4px 12px;font-size:0.8rem;font-weight:600">'
                 f'{pref}{icon} {label}</span>')
nav_html += "</div>"
st.markdown(nav_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 0 — UPLOAD
# ══════════════════════════════════════════════
if cur == 0:
    st.markdown("## 📁 Upload Dataset")
    st.markdown('<div style="color:#475569;font-size:0.9rem;margin-bottom:16px">'
                'Diagnose · Audit · Explain — your complete ML dataset intelligence tool</div>',
                unsafe_allow_html=True)

    uploaded = st.file_uploader("Drop a CSV file here", type=["csv"], label_visibility="collapsed")

    df = None
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            if df.shape[0] < 10 or df.shape[1] < 2:
                st.error("Need at least 10 rows and 2 columns.")
                df = None
            elif df.shape[0] > 500_000:
                df = df.sample(500_000, random_state=42)
                st.warning("Large file — sampled 500,000 rows.")
        except Exception as e:
            st.error(f"Could not read file: {e}")
    elif FALLBACK.exists():
        df = pd.read_csv(FALLBACK)
        st.info("Using built-in sample dataset (stock indicators)")

    if df is not None:
        S["df"] = df
        # Reset downstream state when new file uploaded
        for k in ["analysis_done","report_done","fixed_df","fix_actions",
                  "report_sections","pdf_bytes"]:
            S[k] = False if k in ["analysis_done","report_done"] else (None if k in ["fixed_df","pdf_bytes"] else [])

        k1,k2,k3,k4,k5 = st.columns(5)
        kpi(k1, f"{df.shape[0]:,}", "Rows")
        kpi(k2, str(df.shape[1]), "Columns")
        kpi(k3, f"{df.isna().sum().sum():,}", "Missing")
        kpi(k4, str(df.duplicated().sum()), "Duplicates")
        kpi(k5, str(df.select_dtypes(include=[np.number]).shape[1]), "Numeric Cols")

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([1,1])

        with col_l:
            no_target = st.checkbox("No target column (exploration mode)", value=False)
            if not no_target:
                target = st.selectbox("Target column (what to predict)",
                                       df.columns.tolist(), index=len(df.columns)-1)
                S["target_column"] = target
                task_type, task_reason = detect_task_type(df[target])
                S["task_type"]   = task_type
                S["task_reason"] = task_reason
            else:
                S["target_column"] = df.columns[-1]
                S["task_type"]     = "regression"
                S["task_reason"]   = "Exploration mode — last column used as reference"

        with col_r:
            if not no_target and S["target_column"]:
                color = "#22c55e" if S["task_type"] == "regression" else "#3b82f6"
                st.markdown(
                    f'<div class="card" style="border-color:{color}40;margin-top:28px">'
                    f'<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:1px">Auto-detected</div>'
                    f'<div style="font-size:1rem;font-weight:700;color:{color};margin:4px 0">'
                    f'{S["task_type"].capitalize()}</div>'
                    f'<div style="font-size:0.82rem;color:#64748b">{S["task_reason"]}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("Preview data (first 8 rows)"):
            st.dataframe(df.head(8), use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.button("Run Full Audit →", type="primary", use_container_width=True,
                  on_click=lambda: go_to(1))
    else:
        st.markdown(
            '<div class="card" style="text-align:center;padding:50px;color:#1e2d45">'
            '<div style="font-size:3rem">📂</div>'
            '<div style="margin-top:10px">Upload a CSV file to begin your ML dataset audit</div>'
            '</div>', unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
# PAGE 1 — AUDIT  (runs analysis, shows health)
# ══════════════════════════════════════════════
elif cur == 1:
    df     = S["df"]
    target = S["target_column"]
    task   = S["task_type"]

    if df is None:
        st.warning("Upload a dataset first.")
        st.button("← Go to Upload", on_click=lambda: go_to(0))
        st.stop()

    st.markdown("## 🔍 Dataset Audit")

    if not S["analysis_done"]:
        log_box   = st.empty()
        log_lines: list[str] = []

        def log(msg):
            log_lines.append(msg)
            log_box.markdown(
                "".join(f'<div class="pline">{l}</div>' for l in log_lines[-20:]),
                unsafe_allow_html=True,
            )

        log("🔍 Starting full dataset audit...")

        log("📊 Profiling dataset...")
        S["profile"] = profile_dataset(df, target)
        log(f"  ✅ {S['profile']['rows']:,} rows · {S['profile']['columns']} columns")

        log("🔍 Detecting time-series structure...")
        S["ts_info"] = detect_timeseries(df)
        if S["ts_info"]["is_timeseries"]:
            log(f"  ⏱️ Time-series detected — '{S['ts_info']['datetime_column']}' ({S['ts_info']['frequency_guess']})")
        else:
            log("  ✅ No time-series structure")

        log("🚨 Running leakage detection...")
        S["leakage"] = detect_leakage(df, target)
        if S["leakage"]["leakage_candidates"]:
            log(f"  🚨 Leakage: {S['leakage']['leakage_candidates']}")
        else:
            log("  ✅ No leakage detected")

        log("📈 Analysing distributions...")
        S["distributions"] = analyze_distributions(df, target)
        log(f"  ✅ {len(S['distributions'])} numeric features analysed")

        log("🔗 Detecting feature redundancy...")
        S["redundancy"] = detect_redundancy(df, target)
        log(f"  ✅ {len(S['redundancy']['redundant_pairs'])} redundant pairs found")

        log("❓ Analysing missing value patterns...")
        S["missing_pattern"] = analyze_missing_patterns(df)
        log(f"  ✅ Pattern: {S['missing_pattern']['pattern']}")

        log("🏷️ Inferring column types...")
        S["type_inference"] = infer_column_types(df)
        log(f"  ✅ {S['type_inference']['n_warnings']} type warning(s)")

        log("📏 Checking sample size adequacy...")
        S["sample_check"] = check_sample_size(df, target, task)
        log(f"  ✅ {'Adequate' if S['sample_check']['adequate'] else 'Issues found'}")

        log("⭐ Computing health score...")
        S["health"] = compute_health_score(
            S["profile"], S["leakage"], S["redundancy"],
            S["sample_check"], S["type_inference"],
        )
        log(f"  ✅ Health: {S['health']['total']}/100 — {S['health']['grade']}")

        log("📊 Building scorecard...")
        S["scorecard"] = build_scorecard(
            S["profile"], S["health"], S["redundancy"],
            S["missing_pattern"], S["type_inference"],
            S["sample_check"], S["leakage"],
        )
        log(f"  ✅ Grade: {S['scorecard']['overall_grade']} — {S['scorecard']['overall_verdict']}")

        log("🧹 Applying automated fixes...")
        fix_opts = {
            "drop_constant": True, "drop_id_cols": True,
            "drop_redundant": True, "drop_duplicates": True,
            "impute_missing": True, "cap_outliers": True,
        }
        S["fix_opts"] = fix_opts
        fixed_df, fix_actions = apply_automated_fixes(df, target, S["redundancy"], S["type_inference"], fix_opts)
        S["fixed_df"]    = fixed_df
        S["fix_actions"] = fix_actions
        for a in fix_actions:
            log(f"  ✅ {a}")

        S["analysis_done"] = True
        log("✅ Audit complete!")
        time.sleep(0.3)
        log_box.empty()

    # ── SHOW AUDIT RESULTS ───────────────────
    health    = S["health"]
    scorecard = S["scorecard"]
    profile   = S["profile"]
    total     = health.get("total", 0)
    grade     = health.get("grade", "?")
    gc        = grade_color(grade)

    # Verdict banner
    vclass = "card-green" if total>=75 else ("card-amber" if total>=50 else "card-red")
    st.markdown(
        f'<div class="card {vclass}" style="display:flex;align-items:center;gap:20px;padding:18px 24px">'
        f'<div style="font-size:3rem;font-weight:900;color:{gc};line-height:1">{grade}</div>'
        f'<div>'
        f'<div style="font-size:0.7rem;color:#475569;text-transform:uppercase;letter-spacing:1px">Data Quality Grade</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin:2px 0">{scorecard.get("overall_verdict","")}</div>'
        f'<div style="font-size:0.82rem;color:#475569">{scorecard.get("benchmark","")}</div>'
        f'</div>'
        f'<div style="margin-left:auto;text-align:right">'
        f'<div style="font-size:2rem;font-weight:800;color:{gc}">{total}/100</div>'
        f'<div style="font-size:0.72rem;color:#475569">Health Score</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # Flags row
    flags = []
    if S["leakage"].get("leakage_candidates"):
        flags.append(f'<span class="badge badge-danger">🚨 {len(S["leakage"]["leakage_candidates"])} Leakage Risk</span>')
    if S["ts_info"].get("is_timeseries"):
        flags.append(f'<span class="badge badge-info">⏱️ Time-Series ({S["ts_info"]["frequency_guess"]})</span>')
    if profile.get("missing_pct", 0) > 10:
        flags.append(f'<span class="badge badge-warn">⚠️ {profile["missing_pct"]}% Missing</span>')
    if profile.get("duplicate_rows", 0) > 0:
        flags.append(f'<span class="badge badge-warn">⚠️ {profile["duplicate_rows"]} Duplicates</span>')
    if not S["sample_check"].get("adequate"):
        flags.append('<span class="badge badge-warn">⚠️ Insufficient Samples</span>')
    if S["redundancy"].get("redundant_pairs"):
        flags.append(f'<span class="badge badge-warn">🔗 {len(S["redundancy"]["redundant_pairs"])} Redundant Pairs</span>')
    if flags:
        st.markdown("<div style='margin:8px 0 12px'>" + " ".join(flags) + "</div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # KPIs
    k1,k2,k3,k4,k5,k6 = st.columns(6)
    kpi(k1, f"{profile.get('rows',0):,}", "Rows")
    kpi(k2, str(profile.get('columns',0)), "Columns")
    kpi(k3, f"{profile.get('missing_pct',0)}%", "Missing")
    kpi(k4, str(profile.get('duplicate_rows',0)), "Duplicates")
    kpi(k5, S["task_type"].capitalize(), "Task Type")
    kpi(k6, str(len(S["redundancy"].get("redundant_pairs",[]))), "Redundant Pairs")

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1, 1])

    # Health dimensions
    with col_a:
        section_title("Health Score Breakdown")
        for dim, dv in health.get("dimensions", {}).items():
            pct = int(dv["score"] / max(dv["max"],1) * 100)
            c   = "#22c55e" if pct>=80 else ("#f59e0b" if pct>=50 else "#ef4444")
            st.markdown(
                f'<div style="margin-bottom:10px">'
                f'<div style="display:flex;justify-content:space-between;margin-bottom:3px">'
                f'<span style="font-size:0.83rem;color:#94a3b8">{dim}</span>'
                f'<span style="font-size:0.83rem;color:{c};font-weight:700">{dv["score"]}/{dv["max"]}</span>'
                f'</div>'
                f'{health_bar(dv["score"], dv["max"])}'
                f'<div style="font-size:0.75rem;color:#334155;margin-top:2px">{dv["reason"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # Scorecard sections
    with col_b:
        section_title("Scorecard Detail")
        for sec in scorecard.get("sections", []):
            g  = sec["grade"]
            gc2 = grade_color(g)
            imp_b = "badge-danger" if sec.get("impact")=="CRITICAL" else ("badge-warn" if sec.get("impact")=="HIGH" else "badge-info")
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:10px;padding:8px 10px;'
                f'background:#111827;border-radius:8px;margin-bottom:5px;border:1px solid #1e2d45">'
                f'<div style="font-size:1.2rem;font-weight:800;color:{gc2};width:24px;text-align:center">{g}</div>'
                f'<div style="flex:1">'
                f'<div style="font-size:0.85rem;color:#c0c8d8;font-weight:600">{sec["name"]}</div>'
                f'<div style="font-size:0.75rem;color:#475569">{sec["details"]}</div>'
                f'</div>'
                f'<span class="badge {imp_b}">{sec.get("impact","")}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Fixes applied
    section_title("🧹 Automated Fixes Applied")
    for action in S.get("fix_actions", []):
        st.markdown(
            f'<div style="padding:7px 12px;background:#052e16;border-left:3px solid #166534;'
            f'border-radius:6px;font-size:0.85rem;color:#4ade80;margin-bottom:5px">✅ {action}</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,5])
    c1.button("← Upload", on_click=lambda: go_to(0))
    c2.button("Deep Analysis →", type="primary", on_click=lambda: go_to(2))


# ══════════════════════════════════════════════
# PAGE 2 — DEEP ANALYSIS
# ══════════════════════════════════════════════
elif cur == 2:
    if not S["analysis_done"]:
        st.warning("Run the audit first.")
        st.button("← Go to Audit", on_click=lambda: go_to(1))
        st.stop()

    st.markdown("## 🧬 Deep Analysis")

    tabs = st.tabs(["📈 Distributions", "🔗 Redundancy", "❓ Missing Patterns",
                    "🏷️ Type Inference", "📏 Sample Size", "🚨 Leakage"])

    # ── Distributions ────────────────────────
    with tabs[0]:
        dist = S["distributions"]
        if not dist:
            st.info("No numeric features to analyse.")
        else:
            skewed = {k: v for k, v in dist.items() if abs(v.get("skewness",0)) > 1}
            normal = {k: v for k, v in dist.items() if v.get("is_normal", True)}

            k1, k2, k3 = st.columns(3)
            kpi(k1, str(len(dist)), "Features Analysed")
            kpi(k2, str(len(skewed)), "Skewed (|skew|>1)")
            kpi(k3, str(len(dist)-len(normal)), "Non-Normal")

            st.markdown("<br>", unsafe_allow_html=True)
            section_title("Feature Distribution Summary")

            dist_rows = []
            for col, d in dist.items():
                sk  = d.get("skewness", 0)
                dist_rows.append({
                    "Feature":     col,
                    "Mean":        d.get("mean",""),
                    "Std":         d.get("std",""),
                    "Skewness":    d.get("skewness",""),
                    "Shape":       d.get("shape",""),
                    "Normal?":     "✅" if d.get("is_normal") else "❌",
                    "Recommendation": d.get("recommendation",""),
                })
            dist_df = pd.DataFrame(dist_rows)
            st.dataframe(dist_df, use_container_width=True, hide_index=True)

            # Skewness chart
            if skewed:
                sk_df = pd.DataFrame({
                    "Feature":  list(skewed.keys()),
                    "Skewness": [v["skewness"] for v in skewed.values()],
                }).sort_values("Skewness", key=abs, ascending=False)
                fig = px.bar(sk_df, x="Feature", y="Skewness",
                             color="Skewness", color_continuous_scale="RdBu",
                             color_continuous_midpoint=0, template="plotly_dark",
                             title="Feature Skewness")
                fig.add_hline(y=1.5, line_dash="dash", line_color="#f59e0b",
                              annotation_text="High skew threshold")
                fig.add_hline(y=-1.5, line_dash="dash", line_color="#f59e0b")
                fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                                  coloraxis_showscale=False, margin=dict(t=30))
                st.plotly_chart(fig, use_container_width=True)

    # ── Redundancy ───────────────────────────
    with tabs[1]:
        red  = S["redundancy"]
        pairs = red.get("redundant_pairs", [])
        vif   = red.get("vif_scores", {})

        k1, k2, k3 = st.columns(3)
        kpi(k1, str(len(pairs)), "Redundant Pairs")
        kpi(k2, str(len(red.get("drop_suggestions",[]))), "Suggested Drops")
        kpi(k3, str(len([v for v in vif.values() if v > 5])), "High VIF (>5)")

        st.markdown("<br>", unsafe_allow_html=True)
        col_a, col_b = st.columns([1,1])

        with col_a:
            section_title("Correlation Matrix")
            corr_dict = red.get("corr_matrix", {})
            if corr_dict:
                corr_df = pd.DataFrame(corr_dict)
                fig = px.imshow(corr_df, color_continuous_scale="RdBu",
                                color_continuous_midpoint=0, template="plotly_dark",
                                title="Feature Correlation Heatmap")
                fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                                  margin=dict(t=30), height=400)
                st.plotly_chart(fig, use_container_width=True)

        with col_b:
            section_title("VIF Scores (Multicollinearity)")
            if vif:
                vif_df = pd.DataFrame({"Feature": list(vif.keys()),
                                        "VIF": list(vif.values())}).sort_values("VIF", ascending=False)
                vif_df["Risk"] = vif_df["VIF"].apply(
                    lambda x: "🔴 HIGH" if x>10 else ("🟡 MODERATE" if x>5 else "🟢 OK"))
                st.dataframe(vif_df, use_container_width=True, hide_index=True)
            else:
                st.info("VIF scores not available.")

        if pairs:
            section_title("Redundant Feature Pairs")
            pairs_df = pd.DataFrame([{
                "Feature 1": p["feature_1"], "Feature 2": p["feature_2"],
                "Correlation": p["correlation"], "Severity": p["severity"],
                "Action": p["recommendation"],
            } for p in pairs])
            st.dataframe(pairs_df, use_container_width=True, hide_index=True)

        if red.get("drop_suggestions"):
            st.markdown(
                f'<div class="card card-amber" style="margin-top:10px">'
                f'💡 <b>Suggested drops:</b> {red["drop_suggestions"]} — '
                f'these are captured in your cleaned dataset automatically.</div>',
                unsafe_allow_html=True,
            )

    # ── Missing Patterns ─────────────────────
    with tabs[2]:
        mp = S["missing_pattern"]
        pat = mp.get("pattern", "NONE")
        pat_color = "#22c55e" if pat=="NONE" else ("#f59e0b" if pat=="MCAR" else ("#f97316" if pat=="MAR" else "#ef4444"))

        st.markdown(
            f'<div class="card" style="border-color:{pat_color}40">'
            f'<div style="font-size:0.72rem;color:#475569;text-transform:uppercase">Missing Pattern</div>'
            f'<div style="font-size:1.5rem;font-weight:800;color:{pat_color}">{pat}</div>'
            f'<div style="font-size:0.88rem;color:#94a3b8;margin-top:4px">{mp.get("description","")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<div class="card card-blue" style="margin-top:8px">'
            f'💡 <b>Recommendation:</b> {mp.get("recommendation","")}</div>',
            unsafe_allow_html=True,
        )

        cols_mp = mp.get("columns", {})
        if cols_mp:
            mp_df = pd.DataFrame([{
                "Column": c, "Missing Count": v["count"],
                "Missing %": v["pct"], "Severity": v["severity"],
            } for c, v in cols_mp.items()]).sort_values("Missing %", ascending=False)
            st.dataframe(mp_df, use_container_width=True, hide_index=True)

            # Missing bar chart
            fig = px.bar(mp_df, x="Missing %", y="Column", orientation="h",
                         color="Missing %", color_continuous_scale="Reds",
                         template="plotly_dark", title="Missing Values per Column")
            fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                              coloraxis_showscale=False, margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

        if mp.get("mar_signals"):
            section_title("MAR Signals Detected")
            for sig in mp["mar_signals"]:
                st.markdown(f'<div class="card card-amber" style="padding:8px 12px;font-size:0.85rem">⚠️ {sig}</div>',
                            unsafe_allow_html=True)

    # ── Type Inference ───────────────────────
    with tabs[3]:
        ti = S["type_inference"]
        infs = ti.get("inferences", [])

        k1, k2 = st.columns(2)
        kpi(k1, str(len(infs)), "Columns Checked")
        kpi(k2, str(ti.get("n_warnings",0)), "Type Warnings")

        if ti.get("warnings"):
            st.markdown("<br>", unsafe_allow_html=True)
            for w in ti["warnings"]:
                st.markdown(f'<div class="card card-red" style="padding:8px 12px;font-size:0.85rem">🚨 {w}</div>',
                            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if infs:
            inf_df = pd.DataFrame([{
                "Column": i["column"], "Pandas Type": i["dtype"],
                "Inferred As": i["inferred"], "Status": i["flag"],
                "Action": i["recommendation"],
            } for i in infs])
            st.dataframe(inf_df, use_container_width=True, hide_index=True)

    # ── Sample Size ──────────────────────────
    with tabs[4]:
        sc = S["sample_check"]
        adequate = sc.get("adequate", True)
        sc_color = "#22c55e" if adequate else "#ef4444"

        st.markdown(
            f'<div class="card" style="border-color:{sc_color}40">'
            f'<div style="font-size:1.2rem;font-weight:700;color:{sc_color}">'
            f'{"✅ Sample Size Adequate" if adequate else "❌ Sample Size Insufficient"}</div>'
            f'<div style="font-size:0.85rem;color:#64748b;margin-top:4px">{sc.get("summary","")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        k1, k2, k3 = st.columns(3)
        kpi(k1, f"{sc.get('n_rows',0):,}", "Rows Available")
        kpi(k2, str(sc.get("n_features",0)), "Features")
        kpi(k3, str(len(sc.get("issues",[]))), "Issues Found")

        checks = sc.get("checks", [])
        if checks:
            st.markdown("<br>", unsafe_allow_html=True)
            section_title("Adequacy Checks")
            for c in checks:
                status_color = "#22c55e" if c["pass"] else "#ef4444"
                status_icon  = "✅" if c["pass"] else "❌"
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:12px;padding:9px 12px;'
                    f'background:#111827;border-radius:8px;margin-bottom:5px;border:1px solid #1e2d45">'
                    f'<span style="font-size:1rem">{status_icon}</span>'
                    f'<span style="flex:1;font-size:0.85rem;color:#c0c8d8">{c["rule"]}</span>'
                    f'<span style="font-size:0.8rem;color:#475569">Need {c["needed"]:,} · Have {c["have"]:,}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        if sc.get("projected_improvement"):
            st.markdown(
                f'<div class="card card-blue" style="margin-top:10px">'
                f'📈 <b>More data impact:</b> {sc["projected_improvement"]}</div>',
                unsafe_allow_html=True,
            )

        if sc.get("issues"):
            section_title("Issues Found")
            for issue in sc["issues"]:
                st.markdown(f'<div class="card card-red" style="padding:8px 12px;font-size:0.85rem">❌ {issue}</div>',
                            unsafe_allow_html=True)

    # ── Leakage ──────────────────────────────
    with tabs[5]:
        lk = S["leakage"]
        if lk.get("leakage_candidates"):
            st.error(f"🚨 {len(lk['leakage_candidates'])} potential data leakage feature(s) detected")
            for w in lk.get("warnings",[]):
                st.markdown(f'<div class="card card-red" style="padding:8px 12px;font-size:0.85rem">{w}</div>',
                            unsafe_allow_html=True)
        else:
            st.success("✅ No data leakage detected — all features are within safe correlation bounds.")

        hcf = lk.get("high_correlation_features", {})
        if hcf:
            section_title("All Feature Correlations with Target")
            corr_sorted = sorted(hcf.items(), key=lambda x: abs(x[1]), reverse=True)
            cdf = pd.DataFrame(corr_sorted, columns=["Feature", "Abs Correlation"])
            cdf["Risk"] = cdf["Abs Correlation"].apply(
                lambda x: "🚨 HIGH RISK" if x>0.95 else ("⚠️ Watch" if x>0.85 else "✅ OK"))
            st.dataframe(cdf, use_container_width=True, hide_index=True)

            fig = px.bar(cdf.head(15), x="Abs Correlation", y="Feature", orientation="h",
                         color="Abs Correlation", color_continuous_scale="RdYlGn_r",
                         template="plotly_dark", title="Feature–Target Correlations")
            fig.add_vline(x=0.95, line_dash="dash", line_color="#ef4444",
                          annotation_text="Leakage threshold")
            fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                              coloraxis_showscale=False, margin=dict(t=30))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,5])
    c1.button("← Audit", on_click=lambda: go_to(1))
    c2.button("View Scorecard →", type="primary", on_click=lambda: go_to(3))


# ══════════════════════════════════════════════
# PAGE 3 — SCORECARD
# ══════════════════════════════════════════════
elif cur == 3:
    if not S["analysis_done"]:
        st.warning("Run the audit first.")
        st.button("← Go to Audit", on_click=lambda: go_to(1))
        st.stop()

    st.markdown("## 📊 Data Quality Scorecard")

    scorecard = S["scorecard"]
    profile   = S["profile"]
    grade     = scorecard.get("overall_grade","?")
    gc        = grade_color(grade)

    col_grade, col_details = st.columns([1, 2])

    with col_grade:
        st.markdown(
            f'<div class="card" style="text-align:center;padding:30px 20px">'
            f'<div style="font-size:5rem;font-weight:900;color:{gc};line-height:1">{grade}</div>'
            f'<div style="font-size:0.72rem;color:#475569;text-transform:uppercase;letter-spacing:1px;margin-top:6px">Data Quality Grade</div>'
            f'<div style="font-size:1rem;color:#94a3b8;margin-top:8px">{scorecard.get("overall_verdict","")}</div>'
            f'<div style="font-size:1.5rem;font-weight:700;color:{gc};margin-top:8px">{scorecard.get("overall_score",0)}/100</div>'
            f'<div style="font-size:0.78rem;color:#334155;margin-top:6px">{scorecard.get("benchmark","")}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    with col_details:
        section_title("Section Scores")
        for sec in scorecard.get("sections",[]):
            g2  = sec["grade"]
            gc2 = grade_color(g2)
            pct = sec["score"]
            imp_color = "#ef4444" if sec.get("impact")=="CRITICAL" else ("#f59e0b" if sec.get("impact")=="HIGH" else "#3b82f6")
            st.markdown(
                f'<div style="background:#111827;border:1px solid #1e2d45;border-radius:8px;'
                f'padding:10px 14px;margin-bottom:6px">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:5px">'
                f'<span style="font-size:0.88rem;color:#c0c8d8;font-weight:600">{sec["name"]}</span>'
                f'<span style="display:flex;align-items:center;gap:8px">'
                f'<span style="font-size:0.72rem;color:{imp_color};font-weight:700">{sec.get("impact","")}</span>'
                f'<span style="font-size:1rem;font-weight:800;color:{gc2}">{g2}</span>'
                f'<span style="font-size:0.85rem;color:{gc2}">{sec["score"]}/100</span>'
                f'</span></div>'
                f'{health_bar(sec["score"], 100)}'
                f'<div style="font-size:0.75rem;color:#334155;margin-top:3px">{sec["details"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # Target distribution
    col_a, col_b = st.columns(2)
    with col_a:
        section_title("Target Distribution")
        tgt = S["target_column"]
        if tgt and tgt in S["df"].columns:
            if S["task_type"] == "classification":
                vc = S["df"][tgt].value_counts().reset_index()
                vc.columns = ["Class","Count"]
                fig = px.pie(vc, names="Class", values="Count", template="plotly_dark",
                             color_discrete_sequence=px.colors.sequential.Blues_r)
            else:
                fig = px.histogram(S["df"], x=tgt, template="plotly_dark",
                                   color_discrete_sequence=["#3b82f6"], nbins=40)
            fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                              margin=dict(t=10), height=280)
            st.plotly_chart(fig, use_container_width=True)

    with col_b:
        section_title("Top Feature Correlations")
        top_corr = profile.get("top_correlations",{})
        if top_corr:
            cr_df = pd.DataFrame({"Feature": list(top_corr.keys()),
                                   "Correlation": list(top_corr.values())}).sort_values("Correlation",key=abs,ascending=True)
            fig = px.bar(cr_df, x="Correlation", y="Feature", orientation="h",
                         color="Correlation", color_continuous_scale="RdBu",
                         color_continuous_midpoint=0, template="plotly_dark")
            fig.update_layout(paper_bgcolor="#0a0e1a", plot_bgcolor="#111827",
                              coloraxis_showscale=False, margin=dict(t=10), height=280)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2 = st.columns([1,5])
    c1.button("← Deep Analysis", on_click=lambda: go_to(2))
    c2.button("Generate Expert Report →", type="primary", on_click=lambda: go_to(4))


# ══════════════════════════════════════════════
# PAGE 4 — REPORT
# ══════════════════════════════════════════════
elif cur == 4:
    if not S["analysis_done"]:
        st.warning("Run the audit first.")
        st.button("← Go to Audit", on_click=lambda: go_to(1))
        st.stop()

    st.markdown("## 📄 Expert Report")

    if not S["report_done"]:
        with st.spinner("Groq LLaMA 3.3 70B is writing your expert report..." if S["api_key"]
                        else "Generating rule-based report..."):
            S["report_sections"] = generate_groq_report(
                profile=S["profile"],
                health=S["health"],
                scorecard=S["scorecard"],
                leakage=S["leakage"],
                redundancy=S["redundancy"],
                missing_pattern=S["missing_pattern"],
                distributions=S["distributions"],
                type_inference=S["type_inference"],
                sample_check=S["sample_check"],
                ts_info=S["ts_info"],
                target_column=S["target_column"],
                task_type=S["task_type"],
                api_key=S["api_key"] or None,
            )
            S["pdf_bytes"] = generate_pdf(
                report_sections=S["report_sections"],
                profile=S["profile"],
                health=S["health"],
                scorecard=S["scorecard"],
                leakage=S["leakage"],
                redundancy=S["redundancy"],
                missing_pattern=S["missing_pattern"],
                sample_check=S["sample_check"],
                type_inference=S["type_inference"],
                target_column=S["target_column"],
                task_type=S["task_type"],
                fix_actions=S["fix_actions"],
            )
            S["report_done"] = True
            st.rerun()

    # ── SHOW REPORT ──────────────────────────
    SECTION_TITLES = {
        "executive_summary":        ("📋 Executive Summary", "For management — high-level verdict"),
        "dataset_quality_analysis": ("🔬 Dataset Quality Analysis", "For ML engineers — technical details"),
        "risk_assessment":          ("⚠️ Risk Assessment", "For compliance — what could go wrong"),
        "feature_analysis":         ("🌲 Feature Analysis", "For data scientists — what to transform"),
        "recommended_next_steps":   ("✅ Recommended Next Steps", "Prioritized action list before training"),
        "model_recommendations":    ("🤖 Model Recommendations", "Which algorithm to use and why"),
    }

    # Source badge
    if S["api_key"]:
        st.markdown('<div style="margin-bottom:12px">'
                    '<span class="badge badge-ok">🤖 Written by Groq LLaMA 3.3 70B</span></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown('<div style="margin-bottom:12px">'
                    '<span class="badge badge-warn">⚙️ Rule-based analysis — add Groq API key for AI-written report</span></div>',
                    unsafe_allow_html=True)

    for key, (title, subtitle) in SECTION_TITLES.items():
        text = S["report_sections"].get(key, "")
        if not text:
            continue
        st.markdown(
            f'<div class="rep-section">'
            f'<div class="rep-title">{title}</div>'
            f'<div style="font-size:0.72rem;color:#334155;margin-bottom:8px">{subtitle}</div>'
            f'<div class="rep-body">{text}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    section_title("⬇️ Downloads")
    d1, d2 = st.columns(2)
    with d1:
        if S.get("fixed_df") is not None:
            csv_b = S["fixed_df"].to_csv(index=False).encode()
            st.download_button(
                "📥 Cleaned Dataset (CSV)",
                csv_b, "cleaned_dataset.csv", "text/csv",
                use_container_width=True,
                help="Your original data with all automated fixes applied — ready for model training",
            )
    with d2:
        if S.get("pdf_bytes"):
            st.download_button(
                "📥 Full Audit Report (PDF)",
                S["pdf_bytes"], "automl_audit_report.pdf", "application/pdf",
                use_container_width=True, type="primary",
                help="Professional ML dataset audit report with all findings, grades, and recommendations",
            )

    c1, c2 = st.columns([1,5])
    c1.button("← Scorecard", on_click=lambda: go_to(3))
    if c2.button("🔄 Regenerate Report"):
        S["report_done"] = False
        st.rerun()

st.markdown(
    '<div style="text-align:center;color:#1e2d45;font-size:0.7rem;margin-top:28px">'
    'AutoML Debugger v4.0 · Milestone 1 · Groq LLaMA 3.3 70B · scikit-learn · Streamlit</div>',
    unsafe_allow_html=True,
)
